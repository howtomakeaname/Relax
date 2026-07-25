# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rewards import router as reward_router
from relax.engine.rewards.registry import compute_sync_reward
from relax.engine.rewards.router import _WARNED_ROUTE_ISSUES, resolve_reward_route
from relax.utils.types import Sample


def _make_args(**overrides):
    defaults = {
        "rm_type": None,
        "reward_key": None,
        "reward_router_fallback_rm_type": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(response: str = "", label=None, metadata: dict | None = None):
    return SimpleNamespace(response=response, label=label, metadata=metadata or {})


def _score_for_sample(args, sample):
    route = resolve_reward_route(args, sample)
    if route.zero_score:
        return 0.0
    reward = compute_sync_reward(route.rm_type, sample.response, sample.label, sample.metadata)
    if isinstance(reward, dict):
        return reward["score"]
    return reward


def test_metadata_rm_type_routes_sample():
    sample = _make_sample(metadata={"rm_type": "multiple_choice"})
    route = resolve_reward_route(_make_args(rm_type="dapo"), sample)
    assert route.rm_type == "multiple_choice"
    assert route.reason == "metadata[rm_type]"


def test_format_metadata_math_routes_to_dapo():
    sample = _make_sample(label="9", metadata={"task_type": "math"})
    route = resolve_reward_route(_make_args(), sample)
    assert route.rm_type == "dapo"


def test_explicit_global_rm_type_keeps_legacy_math_reward_name():
    sample = _make_sample(label="9")
    route = resolve_reward_route(_make_args(rm_type="math"), sample)
    assert route.rm_type == "math"
    assert route.reason == "args.rm_type"


def test_label_inference_routes_math_and_multiple_choice():
    math_route = resolve_reward_route(_make_args(), _make_sample(label="9"))
    choice_route = resolve_reward_route(_make_args(), _make_sample(label="<answer>B</answer>"))
    assert math_route.rm_type == "dapo"
    assert choice_route.rm_type == "multiple_choice"


def test_mixed_math_and_multiple_choice_batch_scores_correctly():
    args = _make_args()
    samples = [
        _make_sample(response="Reasoning\nAnswer: 9", label="9"),
        _make_sample(response="<answer>B</answer>", label="<answer>B</answer>"),
    ]
    assert [_score_for_sample(args, sample) for sample in samples] == [1.0, 1.0]


def test_unknown_type_without_fallback_returns_zero_and_warns(monkeypatch):
    _WARNED_ROUTE_ISSUES.clear()
    sample = _make_sample(label="Paris", metadata={"reward_type": "unknown_format"})
    warnings = []
    monkeypatch.setattr(reward_router.logger, "warning", lambda *args, **kwargs: warnings.append(args))

    route = resolve_reward_route(_make_args(), sample)

    assert route.zero_score is True
    assert route.rm_type is None
    assert warnings
    assert "returning zero reward" in warnings[0][2]


def test_unknown_type_uses_configured_fallback(monkeypatch):
    _WARNED_ROUTE_ISSUES.clear()
    sample = _make_sample(label="<answer>A</answer>", metadata={"reward_type": "unknown_format"})
    warnings = []
    monkeypatch.setattr(reward_router.logger, "warning", lambda *args, **kwargs: warnings.append(args))

    route = resolve_reward_route(_make_args(reward_router_fallback_rm_type="multiple_choice"), sample)

    assert route.rm_type == "multiple_choice"
    assert route.fallback_used is True
    assert warnings
    assert "falling back to 'multiple_choice'" in warnings[0][2]


def test_conflicting_metadata_uses_fallback():
    sample = _make_sample(
        label="9",
        metadata={
            "rm_type": "dapo",
            "task_type": "multiple_choice",
        },
    )
    route = resolve_reward_route(_make_args(reward_router_fallback_rm_type="dapo"), sample)
    assert route.rm_type == "dapo"
    assert route.fallback_used is True


@pytest.mark.parametrize(
    ("reward", "reward_key", "expected"),
    [
        (1.0, None, 1.0),
        ({"score": -1.0, "acc": False}, None, -1.0),
        ({"acc_reward": 1.0, "format_reward": 0.0}, "acc_reward", 1.0),
    ],
)
def test_sample_get_reward_value_handles_scalar_and_dict_rewards(reward, reward_key, expected):
    sample = Sample(reward=reward)
    assert sample.get_reward_value(_make_args(reward_key=reward_key)) == expected
