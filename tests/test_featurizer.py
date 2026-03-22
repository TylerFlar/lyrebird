"""Tests for encoder.featurizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from encoder.featurizer import (
    CARD_TYPE_TO_IDX,
    GAME_TO_IDX,
    INPUT_DIM,
    KEYWORD_TO_IDX,
    NUMERIC_EFFECT_KEYS,
    NUMERIC_EFFECT_TO_IDX,
    RARITY_TO_IDX,
    featurize_card,
    featurize_corpus,
)

_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "card_corpus.json"


@pytest.fixture
def corpus():
    with open(_CORPUS_PATH) as f:
        return json.load(f)


class TestConstants:
    def test_input_dim(self):
        expected = len(CARD_TYPE_TO_IDX) + len(RARITY_TO_IDX) + 1 + len(GAME_TO_IDX) + len(NUMERIC_EFFECT_TO_IDX) + len(KEYWORD_TO_IDX)
        assert INPUT_DIM == expected

    def test_no_duplicate_keywords(self):
        assert len(KEYWORD_TO_IDX) == len(set(KEYWORD_TO_IDX.keys()))

    def test_no_duplicate_numeric_keys(self):
        assert len(NUMERIC_EFFECT_KEYS) == len(set(NUMERIC_EFFECT_KEYS))


class TestFeaturizeCard:
    def test_output_shape(self):
        card = {"card_type": "attack", "rarity": "common", "cost": 1, "game": "sts"}
        vec = featurize_card(card)
        assert vec.shape == (INPUT_DIM,)
        assert vec.dtype == np.float32

    def test_card_type_one_hot(self):
        card = {"card_type": "attack", "game": "sts"}
        vec = featurize_card(card)
        ct_start = 0
        ct_end = len(CARD_TYPE_TO_IDX)
        assert vec[ct_start + CARD_TYPE_TO_IDX["attack"]] == 1.0
        assert vec[ct_start:ct_end].sum() == 1.0

    def test_game_one_hot(self):
        card_sts = {"game": "sts"}
        card_bal = {"game": "balatro"}
        vec_sts = featurize_card(card_sts)
        vec_bal = featurize_card(card_bal)
        g_start = len(CARD_TYPE_TO_IDX) + len(RARITY_TO_IDX) + 1
        g_end = g_start + len(GAME_TO_IDX)
        assert vec_sts[g_start + GAME_TO_IDX["sts"]] == 1.0
        assert vec_bal[g_start + GAME_TO_IDX["balatro"]] == 1.0

    def test_cost_normalization(self):
        card = {"cost": 5}
        vec = featurize_card(card)
        cost_idx = len(CARD_TYPE_TO_IDX) + len(RARITY_TO_IDX)
        assert vec[cost_idx] == pytest.approx(0.5)

    def test_numeric_effects_log_scaled(self):
        card = {"numeric_effects": {"damage": 10}}
        vec = featurize_card(card)
        ne_start = len(CARD_TYPE_TO_IDX) + len(RARITY_TO_IDX) + 1 + len(GAME_TO_IDX)
        dmg_idx = ne_start + NUMERIC_EFFECT_TO_IDX["damage"]
        import math
        assert vec[dmg_idx] == pytest.approx(math.log1p(10))

    def test_keywords_multi_hot(self):
        card = {"keywords": ["exhaust", "innate"]}
        vec = featurize_card(card)
        kw_start = INPUT_DIM - len(KEYWORD_TO_IDX)
        assert vec[kw_start + KEYWORD_TO_IDX["exhaust"]] == 1.0
        assert vec[kw_start + KEYWORD_TO_IDX["innate"]] == 1.0
        # Other keywords should be 0
        assert vec[kw_start + KEYWORD_TO_IDX["retain"]] == 0.0

    def test_empty_card(self):
        vec = featurize_card({})
        assert vec.shape == (INPUT_DIM,)
        assert vec.sum() == 0.0

    def test_string_numeric_effect_ignored(self):
        card = {"numeric_effects": {"hand_type": "Flush"}}
        vec = featurize_card(card)
        # Should not crash, and the vector should be all zeros for numeric effects
        ne_start = len(CARD_TYPE_TO_IDX) + len(RARITY_TO_IDX) + 1 + len(GAME_TO_IDX)
        ne_end = ne_start + len(NUMERIC_EFFECT_KEYS)
        assert vec[ne_start:ne_end].sum() == 0.0


class TestFeaturizeCorpus:
    def test_output_shape(self, corpus):
        features = featurize_corpus(corpus)
        assert features.shape == (len(corpus), INPUT_DIM)
        assert features.dtype == np.float32

    def test_no_nans(self, corpus):
        features = featurize_corpus(corpus)
        assert not np.isnan(features).any()

    def test_no_infs(self, corpus):
        features = featurize_corpus(corpus)
        assert not np.isinf(features).any()

    def test_nonzero_features(self, corpus):
        features = featurize_corpus(corpus)
        # Every card should have at least one non-zero feature
        assert (features.sum(axis=1) > 0).all()
