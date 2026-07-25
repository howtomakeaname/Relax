# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any


RewardValue = int | float | dict[str, Any]
SyncRewardHandler = Callable[[str, Any, Mapping[str, Any] | None], RewardValue]

_SYNC_REWARD_HANDLERS: dict[str, SyncRewardHandler] = {}


def normalize_reward_type(rm_type: str) -> str:
    """Return the canonical registry key for a reward type."""
    return rm_type.strip().lower().replace("-", "_").replace(" ", "_")


def register_sync_reward(
    rm_type: str,
    handler: SyncRewardHandler,
    *,
    aliases: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register a synchronous reward handler.

    Args:
        rm_type: Primary reward type name.
        handler: Callable that scores a response.
        aliases: Additional names accepted for the same handler.
        override: Whether an existing registration may be replaced.
    """
    normalized_keys = []
    for key in (rm_type, *aliases):
        normalized = normalize_reward_type(key)
        if normalized in normalized_keys:
            continue
        normalized_keys.append(normalized)

    for normalized in normalized_keys:
        if not override and normalized in _SYNC_REWARD_HANDLERS:
            raise ValueError(f"Sync reward type {normalized!r} is already registered.")
        _SYNC_REWARD_HANDLERS[normalized] = handler


def get_sync_reward_handler(rm_type: str) -> SyncRewardHandler:
    """Return the registered synchronous reward handler."""
    normalized = normalize_reward_type(rm_type)
    try:
        return _SYNC_REWARD_HANDLERS[normalized]
    except KeyError as exc:
        raise NotImplementedError(f"RewardWorker: unknown rm_type={rm_type!r}") from exc


def compute_sync_reward(
    rm_type: str,
    response: str,
    label,
    metadata: Mapping[str, Any] | None = None,
) -> RewardValue:
    """Compute a synchronous reward through the registry."""
    return get_sync_reward_handler(rm_type)(response, label, metadata)


def list_sync_reward_types() -> tuple[str, ...]:
    """Return registered synchronous reward type names."""
    return tuple(sorted(_SYNC_REWARD_HANDLERS))


def _deepscaler_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .deepscaler import get_deepscaler_rule_based_reward

    return get_deepscaler_rule_based_reward(response, label)


def _geo3k_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .geo3k import get_geo3k_reward

    return get_geo3k_reward(response, label)


def _openr1mm_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .openr1mm import get_openr1mm_rule_based_reward

    return get_openr1mm_rule_based_reward(response, label)


def _multiple_choice_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .multiple_choice import get_multiple_choice_reward

    return get_multiple_choice_reward(response, label)


def _dapo_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .math_dapo_utils import compute_score

    return compute_score(response, label)


def _math_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .math_utils import grade_answer_verl

    return 1 if grade_answer_verl(response, label) else 0


def _mopd_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .mopd import get_mopd_reward

    return get_mopd_reward(response, label, dict(metadata or {}))


def _f1_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .f1 import f1_score

    return f1_score(response, label)[0]


def _gpqa_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .gpqa import compute_gpqa_reward

    return compute_gpqa_reward(response, label, metadata=dict(metadata or {}))


def _ifbench_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    from .ifbench import compute_ifbench_reward

    return compute_ifbench_reward(response, label, metadata=dict(metadata or {}))


def _random_reward(response: str, label, metadata: Mapping[str, Any] | None = None) -> RewardValue:
    return random.randint(0, 1)


register_sync_reward("deepscaler", _deepscaler_reward)
register_sync_reward("geo3k", _geo3k_reward, aliases=("geometry3k",))
register_sync_reward("openr1mm", _openr1mm_reward, aliases=("open_r1_mm", "open-r1-mm"))
register_sync_reward("multiple_choice", _multiple_choice_reward, aliases=("multiple-choice", "choice"))
register_sync_reward("dapo", _dapo_reward, aliases=("dapo_math", "dapo-math"))
register_sync_reward("math", _math_reward)
register_sync_reward("mopd", _mopd_reward)
register_sync_reward("f1", _f1_reward)
register_sync_reward("gpqa", _gpqa_reward)
register_sync_reward("ifbench", _ifbench_reward)
register_sync_reward("random", _random_reward)
