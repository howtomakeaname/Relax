# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from relax.utils.logging_utils import get_logger

from .registry import list_sync_reward_types, normalize_reward_type


logger = get_logger(__name__)

ASYNC_REWARD_TYPES = frozenset({"remote_rm", "dapo-genrm", "dapo_genrm", "dummy"})
EXPLICIT_METADATA_KEYS = ("rm_type",)
FORMAT_METADATA_KEYS = ("reward_type", "reward_format", "task_type", "format")
TYPE_METADATA_KEYS = EXPLICIT_METADATA_KEYS + FORMAT_METADATA_KEYS

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUMERIC_LABEL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_MATH_MARKERS = ("\\frac", "\\sqrt", "\\boxed", "\\pi", "^", "_", "{", "}")
_WARNED_ROUTE_ISSUES: set[tuple[str, str | None]] = set()


@dataclass(frozen=True)
class RewardRoute:
    """Resolved reward route for one sample."""

    rm_type: str | None
    reason: str
    fallback_used: bool = False
    zero_score: bool = False


def canonicalize_reward_type(value: Any, *, metadata_key: str | None = None) -> str | None:
    """Canonicalize a reward type token from CLI or metadata."""
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    boxed = False
    normalized = normalize_reward_type(raw)
    if normalized.startswith("boxed_"):
        boxed = True
        normalized = normalized[len("boxed_") :]

    aliases = {
        "multiplechoice": "multiple_choice",
        "multiple_choice": "multiple_choice",
        "choice": "multiple_choice",
        "choices": "multiple_choice",
        "mcq": "multiple_choice",
        "dapo_math": "dapo",
        "dapo": "dapo",
        "remote_rm": "remote_rm",
        "dummy": "dummy",
        "dapo_genrm": "dapo-genrm",
        "open_r1_mm": "openr1mm",
        "openr1mm": "openr1mm",
        "geo3k": "geo3k",
        "geometry3k": "geo3k",
    }
    if metadata_key in FORMAT_METADATA_KEYS:
        aliases |= {
            "math": "dapo",
            "mathematics": "dapo",
            "math_dapo": "dapo",
        }
    else:
        aliases["math"] = "math"

    canonical = aliases.get(normalized, normalized)
    return f"boxed_{canonical}" if boxed else canonical


def is_known_reward_type(rm_type: str | None) -> bool:
    """Return whether a reward type can be executed by the built-in router."""
    if not rm_type:
        return False

    normalized = rm_type
    if normalized.startswith("boxed_"):
        normalized = normalized[len("boxed_") :]

    return normalized in ASYNC_REWARD_TYPES or normalize_reward_type(normalized) in set(list_sync_reward_types())


def resolve_reward_route(args: Any, sample: Any) -> RewardRoute:
    """Resolve the reward type for a sample.

    Priority:
      1. sample metadata reward type fields
      2. global --rm-type
      3. metadata/label inference
      4. configured fallback
      5. zero score
    """
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}

    metadata_route = _resolve_metadata_route(args, metadata)
    if metadata_route is not None:
        return metadata_route

    args_rm_type = canonicalize_reward_type(getattr(args, "rm_type", None), metadata_key="rm_type")
    if args_rm_type:
        if is_known_reward_type(args_rm_type):
            return RewardRoute(rm_type=args_rm_type, reason="args.rm_type")
        return _fallback_or_zero(
            args,
            reason=f"unknown args rm_type {args_rm_type!r}",
            warning_key=args_rm_type,
            metadata=metadata,
            label=getattr(sample, "label", None),
        )

    inferred = _infer_reward_type(metadata=metadata, label=getattr(sample, "label", None))
    if inferred:
        return RewardRoute(rm_type=inferred, reason="metadata/label inference")

    return _fallback_or_zero(
        args,
        reason="missing reward type",
        warning_key=None,
        metadata=metadata,
        label=getattr(sample, "label", None),
    )


def _resolve_metadata_route(args: Any, metadata: Mapping[str, Any]) -> RewardRoute | None:
    declared: list[tuple[str, str]] = []
    for key in TYPE_METADATA_KEYS:
        if key not in metadata:
            continue
        canonical = canonicalize_reward_type(metadata.get(key), metadata_key=key)
        if canonical:
            declared.append((key, canonical))

    if not declared:
        return None

    types = {rm_type for _, rm_type in declared}
    if len(types) > 1:
        detail = ", ".join(f"{key}={rm_type}" for key, rm_type in declared)
        return _fallback_or_zero(
            args,
            reason=f"conflicting reward metadata ({detail})",
            warning_key=detail,
            metadata=metadata,
            label=None,
        )

    rm_type = declared[0][1]
    if is_known_reward_type(rm_type):
        keys = ",".join(key for key, _ in declared)
        return RewardRoute(rm_type=rm_type, reason=f"metadata[{keys}]")

    return _fallback_or_zero(
        args,
        reason=f"unknown metadata reward type {rm_type!r}",
        warning_key=rm_type,
        metadata=metadata,
        label=None,
    )


def _infer_reward_type(metadata: Mapping[str, Any], label: Any) -> str | None:
    data_source = _metadata_text(metadata, "data_source", "source", "dataset", "task")
    if data_source:
        if any(token in data_source for token in ("multiple_choice", "multiple-choice", "choice", "nextqa", "avqa")):
            return "multiple_choice"
        if "gpqa" in data_source:
            return "gpqa"
        if "openr1mm" in data_source or "multimodal-open-r1" in data_source:
            return "openr1mm"
        if "geo3k" in data_source or "geometry3k" in data_source:
            return "geo3k"
        if "dapo" in data_source or "math" in data_source:
            return "dapo"

    if any(key in metadata for key in ("choices", "valid_letters", "correct_letter")):
        return "multiple_choice"

    label_text = _label_text(label)
    if label_text is None:
        return None

    if _ANSWER_TAG_RE.search(label_text):
        return "multiple_choice"
    if _NUMERIC_LABEL_RE.fullmatch(label_text.strip()):
        return "dapo"
    if any(marker in label_text for marker in _MATH_MARKERS):
        return "dapo"

    return None


def _metadata_text(metadata: Mapping[str, Any], *keys: str) -> str:
    values = []
    for key in keys:
        value = metadata.get(key)
        if value:
            values.append(str(value).strip().lower())
    return " ".join(values)


def _label_text(label: Any) -> str | None:
    if label is None:
        return None
    if isinstance(label, Mapping):
        for key in ("ground_truth", "answer", "label", "correct_answer"):
            value = label.get(key)
            if value is not None:
                return str(value)
        return None
    return str(label)


def _fallback_or_zero(
    args: Any,
    *,
    reason: str,
    warning_key: str | None,
    metadata: Mapping[str, Any],
    label: Any,
) -> RewardRoute:
    fallback = canonicalize_reward_type(getattr(args, "reward_router_fallback_rm_type", None), metadata_key="rm_type")
    if fallback and is_known_reward_type(fallback):
        _warn_once(
            reason=reason,
            warning_key=warning_key,
            metadata=metadata,
            label=label,
            action=f"falling back to {fallback!r}",
        )
        return RewardRoute(rm_type=fallback, reason=reason, fallback_used=True)

    _warn_once(
        reason=reason,
        warning_key=warning_key,
        metadata=metadata,
        label=label,
        action="returning zero reward",
    )
    return RewardRoute(rm_type=None, reason=reason, zero_score=True)


def _warn_once(
    *,
    reason: str,
    warning_key: str | None,
    metadata: Mapping[str, Any],
    label: Any,
    action: str,
) -> None:
    key = (reason, warning_key)
    if key in _WARNED_ROUTE_ISSUES:
        return
    _WARNED_ROUTE_ISSUES.add(key)

    label_preview = str(label)[:120] if label is not None else None
    logger.warning(
        "Reward router %s; %s. metadata_keys=%s label_preview=%r",
        reason,
        action,
        sorted(metadata.keys()),
        label_preview,
    )
