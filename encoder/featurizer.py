"""Convert card_corpus.json entries into fixed-size feature vectors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── Card type one-hot (9 dims) ──────────────────────────────────────────────
CARD_TYPES = [
    "attack",
    "curse",
    "defense",
    "modifier",
    "playing_card",
    "power",
    "scaling",
    "status",
    "utility",
]
CARD_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(CARD_TYPES)}

# ── Rarity one-hot (7 dims) ─────────────────────────────────────────────────
RARITIES = [
    "basic",
    "common",
    "curse",
    "legendary",
    "rare",
    "special",
    "uncommon",
]
RARITY_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(RARITIES)}

# ── Game one-hot (2 dims) ───────────────────────────────────────────────────
GAMES = ["balatro", "sts"]
GAME_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GAMES)}

# ── Numeric effects (30 dims) — only numeric-valued keys ────────────────────
NUMERIC_EFFECT_KEYS = [
    "block",
    "block_upgrade",
    "chip_mod",
    "chips",
    "d_remaining",
    "damage",
    "damage_upgrade",
    "destroy",
    "discard_sub",
    "discards",
    "every",
    "extra",
    "faces",
    "h_mod",
    "h_plays",
    "hand_add",
    "hand_size",
    "increase",
    "magic_number",
    "magic_number_upgrade",
    "max",
    "max_highlighted",
    "min",
    "money",
    "mult",
    "mult_flat",
    "mult_x",
    "odds",
    "size",
    "xmult",
]
NUMERIC_EFFECT_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(NUMERIC_EFFECT_KEYS)}

# ── Keywords multi-hot (68 dims) ────────────────────────────────────────────
KEYWORDS = [
    "1_in_10_mult",
    "ace_buff",
    "all_face_cards",
    "bonus_dollars",
    "bonus_rerolls",
    "can_be_eternal",
    "can_perish",
    "card_conversion",
    "card_mult",
    "card_removal",
    "copyable",
    "copycat",
    "credit",
    "disable_blind_effect",
    "discard_chips",
    "discard_dollars",
    "discard_size",
    "dollar_doubler",
    "dollars_for_gold_cards",
    "enhance",
    "ethereal",
    "even_card_buff",
    "exhaust",
    "face_card_dollar_chance",
    "face_card_double",
    "glass_card",
    "hand_card_double",
    "hand_played_mult",
    "hand_size",
    "hand_size,_plays",
    "hand_size_mult",
    "hand_upgrade",
    "innate",
    "jack_discard_effect",
    "joker_mult",
    "joker_payout",
    "low_card_double",
    "mult",
    "multi_target",
    "no_discard_mult",
    "odd_card_buff",
    "prevent_death",
    "random_joker",
    "random_mult",
    "retain",
    "round_bonus",
    "scary_face_cards",
    "set_mult",
    "shop_size",
    "socialized_mult",
    "spawn_tarot",
    "steel_card_buff",
    "stone_card_buff",
    "stone_card_hands",
    "suit_conversion",
    "suit_mult",
    "targets_all_enemy",
    "targets_enemy",
    "targets_none",
    "targets_self",
    "tarot_buff",
    "type_mult",
    "unlocker",
    "upgrade_hand_chance",
    "x1.5_mult",
    "x1.5_mult_club_7",
    "x2_mult",
    "x3_mult",
]
KEYWORD_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(KEYWORDS)}

# ── Derived constants ────────────────────────────────────────────────────────
_CARD_TYPE_DIM = len(CARD_TYPES)        # 9
_RARITY_DIM = len(RARITIES)             # 7
_COST_DIM = 1                           # 1
_GAME_DIM = len(GAMES)                  # 2
_NUMERIC_DIM = len(NUMERIC_EFFECT_KEYS) # 30
_KEYWORD_DIM = len(KEYWORDS)            # 68

INPUT_DIM = _CARD_TYPE_DIM + _RARITY_DIM + _COST_DIM + _GAME_DIM + _NUMERIC_DIM + _KEYWORD_DIM  # 117


def _log_scale(v: float) -> float:
    """sign(v) * log(1 + |v|)"""
    return math.copysign(math.log1p(abs(v)), v)


def featurize_card(card: dict[str, Any]) -> np.ndarray:
    """Convert a single card_corpus.json entry to a fixed-size feature vector.

    Returns shape ``(INPUT_DIM,)`` float32.
    """
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    offset = 0

    # Card type one-hot
    ct_idx = CARD_TYPE_TO_IDX.get(card.get("card_type", ""), -1)
    if ct_idx >= 0:
        vec[offset + ct_idx] = 1.0
    offset += _CARD_TYPE_DIM

    # Rarity one-hot
    r_idx = RARITY_TO_IDX.get(card.get("rarity", ""), -1)
    if r_idx >= 0:
        vec[offset + r_idx] = 1.0
    offset += _RARITY_DIM

    # Cost (normalized by 10)
    cost = card.get("cost")
    if cost is not None and isinstance(cost, (int, float)):
        vec[offset] = float(cost) / 10.0
    offset += _COST_DIM

    # Game one-hot
    g_idx = GAME_TO_IDX.get(card.get("game", ""), -1)
    if g_idx >= 0:
        vec[offset + g_idx] = 1.0
    offset += _GAME_DIM

    # Numeric effects (log-scaled)
    effects = card.get("numeric_effects", {})
    for key, idx in NUMERIC_EFFECT_TO_IDX.items():
        val = effects.get(key)
        if val is not None and isinstance(val, (int, float)):
            vec[offset + idx] = _log_scale(float(val))
    offset += _NUMERIC_DIM

    # Keywords multi-hot
    for kw in card.get("keywords", []):
        kw_idx = KEYWORD_TO_IDX.get(kw, -1)
        if kw_idx >= 0:
            vec[offset + kw_idx] = 1.0

    return vec


def featurize_corpus(corpus: list[dict[str, Any]]) -> np.ndarray:
    """Featurize every card in the corpus.

    Returns shape ``(N, INPUT_DIM)`` float32.
    """
    return np.stack([featurize_card(c) for c in corpus], axis=0)
