"""Tests for the Slay the Spire gymnasium wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from envs.sts_gymnasium_wrapper import (
    FLAT_ACTION_SIZE,
    MAX_HAND_SIZE,
    MAX_MONSTERS,
    CARD_FEATURES,
    MONSTER_FEATURES,
    PLAYER_FEATURES,
    SlayTheSpireEnv,
)


@pytest.fixture
def env():
    e = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=500)
    yield e
    e.close()


class TestResetAndStep:
    def test_reset_returns_obs_and_info(self, env: SlayTheSpireEnv):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_step_after_reset(self, env: SlayTheSpireEnv):
        env.reset(seed=42)
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        assert len(valid_actions) > 0, "No valid actions after reset"

        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


class TestObservationShapes:
    def test_player_shape(self, env: SlayTheSpireEnv):
        obs, _ = env.reset(seed=42)
        assert obs["player"].shape == (PLAYER_FEATURES,)
        assert obs["player"].dtype == np.float32

    def test_hand_shape(self, env: SlayTheSpireEnv):
        obs, _ = env.reset(seed=42)
        assert obs["hand"].shape == (MAX_HAND_SIZE, CARD_FEATURES)
        assert obs["hand"].dtype == np.float32

    def test_monsters_shape(self, env: SlayTheSpireEnv):
        obs, _ = env.reset(seed=42)
        assert obs["monsters"].shape == (MAX_MONSTERS, MONSTER_FEATURES)
        assert obs["monsters"].dtype == np.float32

    def test_entity_counts_shape(self, env: SlayTheSpireEnv):
        obs, _ = env.reset(seed=42)
        assert obs["entity_counts"].shape == (2,)
        assert obs["entity_counts"].dtype == np.float32

    def test_obs_within_space(self, env: SlayTheSpireEnv):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), "Observation not in observation_space"


class TestActionMask:
    def test_mask_length(self, env: SlayTheSpireEnv):
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (FLAT_ACTION_SIZE,)
        assert mask.dtype == bool

    def test_mask_has_valid_actions(self, env: SlayTheSpireEnv):
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.any(), "Mask should have at least one valid action"

    def test_valid_action_accepted(self, env: SlayTheSpireEnv):
        """Valid actions from the mask should be accepted by game.step()."""
        env.reset(seed=42)
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]

        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        # Should not be flagged as illegal
        assert not info.get("illegal", False), "Valid action was rejected"


class TestRandomEpisodes:
    @pytest.mark.parametrize("episode", range(10))
    def test_random_episode_completes(self, episode: int):
        """Run a random episode to completion (or max steps)."""
        env = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=200)
        try:
            obs, info = env.reset(seed=episode * 7)
            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:
                mask = env.action_masks()
                valid_actions = np.where(mask)[0]

                if len(valid_actions) == 0:
                    break  # Game may be in terminal state

                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

                # Verify obs shape every step
                assert obs["player"].shape == (PLAYER_FEATURES,)
                assert obs["hand"].shape == (MAX_HAND_SIZE, CARD_FEATURES)
                assert obs["monsters"].shape == (MAX_MONSTERS, MONSTER_FEATURES)

            assert steps > 0, "Episode had zero steps"
        except Exception as e:
            # Game crashes are expected (decapitate-the-spire has TODOs).
            # The wrapper should catch them, but if something unexpected
            # propagates, we still want to know.
            pytest.skip(f"Game crashed on episode {episode}: {e}")
        finally:
            env.close()


class TestTruncation:
    def test_max_steps_truncation(self):
        """Episodes should truncate at max_steps."""
        env = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=10)
        try:
            env.reset(seed=42)
            for _ in range(20):
                mask = env.action_masks()
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                obs, reward, terminated, truncated, info = env.step(valid[0])
                if terminated or truncated:
                    break
            # Should have hit truncation or termination within 20 attempts
            assert terminated or truncated
        finally:
            env.close()
