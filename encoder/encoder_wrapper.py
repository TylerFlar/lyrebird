"""Gymnasium ObservationWrapper that replaces raw card features with encoder embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from encoder.card_encoder import EMBED_DIM, CardEncoder
from encoder.featurizer import INPUT_DIM, featurize_card

logger = logging.getLogger(__name__)

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_PATH = _PROJECT_ROOT / "data" / "card_corpus.json"


class EncoderObsWrapper(gym.ObservationWrapper):
    """Replace per-card feature vectors in StS observations with learned embeddings.

    The underlying ``SlayTheSpireEnv`` produces ``hand`` with shape
    ``(MAX_HAND_SIZE, 8)``.  This wrapper substitutes each card's 8-dim
    feature row with the encoder's ``embed_dim``-dim output, looked up from
    a pre-computed table keyed by card class name.

    Cards not found in the corpus (shouldn't happen if corpus is complete)
    get a zero embedding.
    """

    def __init__(
        self,
        env: gym.Env,
        encoder_path: str | Path,
        game: str = "sts",
        freeze_encoder: bool = True,
        corpus_path: str | Path | None = None,
    ):
        super().__init__(env)

        self.game = game
        self.embed_dim = EMBED_DIM

        # Load encoder
        self.encoder = CardEncoder(INPUT_DIM, embed_dim=EMBED_DIM)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        self.encoder.eval()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # Build card class_name -> embedding lookup
        corpus_path = Path(corpus_path) if corpus_path else _CORPUS_PATH
        with open(corpus_path) as f:
            corpus = json.load(f)

        self._card_embeddings: dict[str, np.ndarray] = {}
        self._zero_embedding = np.zeros(EMBED_DIM, dtype=np.float32)

        for card_data in corpus:
            if card_data.get("game") != game:
                continue
            class_name = card_data.get("raw", {}).get("class_name")
            if not class_name:
                continue
            features = featurize_card(card_data)
            with torch.no_grad():
                emb = self.encoder(torch.tensor(features).unsqueeze(0)).squeeze(0)
            self._card_embeddings[class_name] = emb.numpy()

        logger.info("Loaded %d card embeddings for game=%s", len(self._card_embeddings), game)

        # Modify observation space: hand goes from (MAX_HAND, old_card_dim) to (MAX_HAND, embed_dim)
        old_spaces = dict(self.env.observation_space.spaces)
        old_hand = old_spaces["hand"]
        max_hand = old_hand.shape[0]
        old_spaces["hand"] = spaces.Box(
            low=0.0, high=1.0, shape=(max_hand, EMBED_DIM), dtype=np.float32,
        )
        self.observation_space = spaces.Dict(old_spaces)

        # Import dg lazily to get card class names at runtime
        self._max_hand = max_hand

    def action_masks(self) -> np.ndarray:
        """Forward action_masks() to the underlying env for MaskablePPO compatibility."""
        return self.env.action_masks()

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = dict(obs)  # shallow copy
        new_hand = np.zeros((self._max_hand, self.embed_dim), dtype=np.float32)

        # Access the underlying game's hand to get card class names
        game = self.unwrapped._game
        if game is not None:
            room = game.ctx.d.get_curr_room()
            # Import RoomPhase from dts
            import decapitate_the_spire.game as dg

            if room.phase == dg.RoomPhase.COMBAT:
                for idx, card in enumerate(game.ctx.player.hand):
                    if idx >= self._max_hand:
                        break
                    class_name = card.__class__.__name__
                    emb = self._card_embeddings.get(class_name)
                    if emb is not None:
                        new_hand[idx] = emb
                    # else: stays zero (unknown card)

        obs["hand"] = new_hand
        return obs
