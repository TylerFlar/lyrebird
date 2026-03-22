"""Tests for encoder.encoder_wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from encoder.card_encoder import EMBED_DIM, CardEncoder
from encoder.encoder_wrapper import EncoderObsWrapper
from encoder.featurizer import INPUT_DIM
from envs.sts_gymnasium_wrapper import MAX_HAND_SIZE, SlayTheSpireEnv


@pytest.fixture
def encoder_path():
    """Create a temporary encoder weights file."""
    encoder = CardEncoder(INPUT_DIM, embed_dim=EMBED_DIM)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_encoder.pt"
        torch.save(encoder.state_dict(), str(path))
        yield path


@pytest.fixture
def wrapped_env(encoder_path):
    env = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=200)
    wrapped = EncoderObsWrapper(env, encoder_path=encoder_path, game="sts")
    yield wrapped
    wrapped.close()


class TestObservationSpace:
    def test_hand_shape_updated(self, wrapped_env):
        hand_space = wrapped_env.observation_space["hand"]
        assert hand_space.shape == (MAX_HAND_SIZE, EMBED_DIM)

    def test_player_space_unchanged(self, wrapped_env):
        player_space = wrapped_env.observation_space["player"]
        assert player_space.shape == wrapped_env.unwrapped.observation_space["player"].shape

    def test_monsters_space_unchanged(self, wrapped_env):
        monster_space = wrapped_env.observation_space["monsters"]
        assert monster_space.shape == wrapped_env.unwrapped.observation_space["monsters"].shape


class TestResetAndStep:
    def test_reset_returns_correct_shapes(self, wrapped_env):
        obs, info = wrapped_env.reset(seed=42)
        assert obs["hand"].shape == (MAX_HAND_SIZE, EMBED_DIM)
        assert obs["hand"].dtype == np.float32

    def test_step_returns_correct_shapes(self, wrapped_env):
        wrapped_env.reset(seed=42)
        mask = wrapped_env.action_masks()
        valid = np.where(mask)[0]
        assert len(valid) > 0

        obs, reward, term, trunc, info = wrapped_env.step(valid[0])
        assert obs["hand"].shape == (MAX_HAND_SIZE, EMBED_DIM)

    def test_random_episode(self, wrapped_env):
        """Run a few steps of a random episode through the wrapped env."""
        obs, _ = wrapped_env.reset(seed=42)
        for _ in range(20):
            mask = wrapped_env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, reward, term, trunc, info = wrapped_env.step(np.random.choice(valid))
            assert obs["hand"].shape == (MAX_HAND_SIZE, EMBED_DIM)
            if term or trunc:
                break
