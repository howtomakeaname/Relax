# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.engine.rewards.registry import (
    compute_sync_reward,
    get_sync_reward_handler,
    list_sync_reward_types,
    register_sync_reward,
)


def test_builtin_rewards_are_registered():
    reward_types = set(list_sync_reward_types())
    assert "dapo" in reward_types
    assert "multiple_choice" in reward_types
    assert "openr1mm" in reward_types


def test_new_reward_registration_is_one_line():
    def custom_reward(response, label, metadata=None):
        return 1.0 if response == label else 0.0

    register_sync_reward("unit_test_exact_match_reward", custom_reward)

    handler = get_sync_reward_handler("unit_test_exact_match_reward")
    assert handler("A", "A", None) == 1.0
    assert compute_sync_reward("unit_test_exact_match_reward", "A", "B") == 0.0


def test_duplicate_reward_registration_is_rejected():
    def custom_reward(response, label, metadata=None):
        return 0.0

    register_sync_reward("unit_test_duplicate_reward", custom_reward)
    with pytest.raises(ValueError, match="already registered"):
        register_sync_reward("unit_test_duplicate_reward", custom_reward)


def test_unknown_reward_type_raises_clear_error():
    with pytest.raises(NotImplementedError, match="unknown rm_type"):
        get_sync_reward_handler("not_registered_reward")
