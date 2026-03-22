"""Gymnasium wrapper for decapitate-the-spire (Slay the Spire: Exordium / Silent)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure decapitate-the-spire is importable from sibling directory.
_DTS_ROOT = Path(__file__).resolve().parents[2] / "decapitate-the-spire"
if str(_DTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DTS_ROOT))

from decapitate_the_spire import game as dg  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from decapitate-the-spire
# ---------------------------------------------------------------------------
ACTION_0_LEN: int = 1 + dg.MAX_HAND_SIZE + 2 * dg.MAX_POTION_SLOTS  # 21
ACTION_1_LEN: int = 1 + dg.MAX_NUM_MONSTERS_IN_GROUP  # 6
FLAT_ACTION_SIZE: int = ACTION_0_LEN * ACTION_1_LEN  # 126

MAX_HAND_SIZE: int = dg.MAX_HAND_SIZE  # 10
MAX_MONSTERS: int = dg.MAX_NUM_MONSTERS_IN_GROUP  # 5

# Feature counts
PLAYER_FEATURES = 10
CARD_FEATURES = 8  # cost + 5 type one-hot + damage + block
MONSTER_FEATURES = 8  # hp_ratio + abs_hp + block + 4 intent one-hot + move_damage

# Intent grouping: map the 12 Intent variants into 4 categories
_INTENT_ATTACK = {dg.Intent.ATTACK, dg.Intent.ATTACK_DEBUFF, dg.Intent.ATTACK_DEFEND, dg.Intent.ATTACK_BUFF}
_INTENT_DEFEND = {dg.Intent.DEFEND, dg.Intent.DEFEND_BUFF}
_INTENT_BUFF = {dg.Intent.BUFF, dg.Intent.DEBUFF}
# Everything else (UNKNOWN, ESCAPE, STUN, SLEEP) -> "other"


def _intent_one_hot(intent: Optional[dg.Intent]) -> list[float]:
    """Return 4-element one-hot: [attack, defend, buff, other]."""
    if intent is None:
        return [0.0, 0.0, 0.0, 1.0]
    if intent in _INTENT_ATTACK:
        return [1.0, 0.0, 0.0, 0.0]
    if intent in _INTENT_DEFEND:
        return [0.0, 1.0, 0.0, 0.0]
    if intent in _INTENT_BUFF:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


# CardType one-hot order: ATTACK=0, SKILL=1, POWER=2, STATUS=3, CURSE=4
_CARD_TYPE_INDEX = {
    dg.CardType.ATTACK: 0,
    dg.CardType.SKILL: 1,
    dg.CardType.POWER: 2,
    dg.CardType.STATUS: 3,
    dg.CardType.CURSE: 4,
}


def _card_type_one_hot(card_type: dg.CardType) -> list[float]:
    vec = [0.0] * 5
    vec[_CARD_TYPE_INDEX.get(card_type, 4)] = 1.0
    return vec


class SlayTheSpireEnv(gym.Env):
    """Gymnasium environment wrapping decapitate-the-spire.

    Action space: Discrete(126) — flattened (21 × 6) action grid.
    Observation space: Dict with player, hand, monsters, entity_counts.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_hp: int = 80,
        energy_per_turn: int = 3,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.max_hp = max_hp
        self.energy_per_turn = energy_per_turn
        self.max_steps = max_steps

        # Action space: flat Discrete over the 21×6 grid
        self.action_space = spaces.Discrete(FLAT_ACTION_SIZE)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(low=0.0, high=1.0, shape=(PLAYER_FEATURES,), dtype=np.float32),
                "hand": spaces.Box(low=0.0, high=1.0, shape=(MAX_HAND_SIZE, CARD_FEATURES), dtype=np.float32),
                "monsters": spaces.Box(low=0.0, high=1.0, shape=(MAX_MONSTERS, MONSTER_FEATURES), dtype=np.float32),
                "entity_counts": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

        # Internal state
        self._game: Optional[dg.Game] = None
        self._steps: int = 0
        self._prev_floor: int = 0
        self._prev_player_hp: int = 0
        self._prev_monster_hp_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        max_hp = self.max_hp
        energy = self.energy_per_turn

        def create_player(ctx: dg.CCG.Context) -> dg.Player:
            return dg.TheSilent(ctx, max_hp, energy)

        self._game = dg.Game(create_player, create_dungeon=dg.Exordium)
        self._steps = 0
        self._prev_floor = self._game.ctx.d.floor_num
        self._prev_player_hp = self._game.ctx.player.current_health
        self._prev_monster_hp_ratio = self._get_total_monster_hp_ratio()

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._game is not None, "Call reset() before step()"

        # Unflatten action
        action_0 = action // ACTION_1_LEN
        action_1 = action % ACTION_1_LEN
        action_coord = (action_0, action_1)

        self._steps += 1

        # Step the game (with crash protection)
        try:
            raw_reward, is_terminal, game_info = self._game.step(action_coord)
        except Exception as e:
            logger.warning("Game crashed: %s", e)
            obs = self._build_obs()
            info = self._build_info()
            info["crash"] = str(e)
            return obs, -1.0, True, False, info

        # Compute shaped reward
        reward = self._shape_reward(raw_reward, is_terminal, game_info)

        # Update tracking state
        self._prev_floor = self._game.ctx.d.floor_num
        self._prev_player_hp = self._game.ctx.player.current_health
        self._prev_monster_hp_ratio = self._get_total_monster_hp_ratio()

        # Truncation check
        truncated = not is_terminal and self._steps >= self.max_steps

        obs = self._build_obs()
        info = self._build_info()
        info.update(game_info)
        if is_terminal:
            info["floor_reached"] = self._game.ctx.d.floor_num

        return obs, reward, is_terminal, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return flat bool mask of length 126 for MaskablePPO."""
        if self._game is None:
            return np.zeros(FLAT_ACTION_SIZE, dtype=bool)

        raw_mask = self._game.generate_action_mask()
        flat = np.zeros(FLAT_ACTION_SIZE, dtype=bool)
        for i in range(ACTION_0_LEN):
            for j in range(ACTION_1_LEN):
                flat[i * ACTION_1_LEN + j] = raw_mask[i][j]
        return flat

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        player = self._game.ctx.player
        dungeon = self._game.ctx.d
        room = dungeon.get_curr_room()
        in_combat = room.phase == dg.RoomPhase.COMBAT

        # --- Player features ---
        player_obs = np.zeros(PLAYER_FEATURES, dtype=np.float32)
        player_obs[0] = player.current_health / max(player.max_health, 1)
        player_obs[1] = min(player.current_block / 100.0, 1.0)
        player_obs[2] = player.energy_manager.player_current_energy / max(self.energy_per_turn, 1)
        player_obs[3] = min(dungeon.floor_num / 50.0, 1.0)
        player_obs[4] = min(len(player.draw_pile) / 30.0, 1.0)
        player_obs[5] = min(len(player.discard_pile) / 30.0, 1.0)
        player_obs[6] = min(len(player.exhaust_pile) / 10.0, 1.0)
        player_obs[7] = min(len(player.relics) / 10.0, 1.0)
        player_obs[8] = min(len(player.potions) / 5.0, 1.0)
        player_obs[9] = 1.0 if in_combat else 0.0

        # --- Hand cards ---
        hand_obs = np.zeros((MAX_HAND_SIZE, CARD_FEATURES), dtype=np.float32)
        if in_combat:
            for idx, card in enumerate(player.hand):
                if idx >= MAX_HAND_SIZE:
                    break
                hand_obs[idx] = self._encode_card(card)

        # --- Monsters ---
        monster_obs = np.zeros((MAX_MONSTERS, MONSTER_FEATURES), dtype=np.float32)
        num_monsters = 0
        if in_combat and room.monster_group is not None:
            for idx, monster in enumerate(room.monster_group):
                if idx >= MAX_MONSTERS:
                    break
                if monster.is_dead or monster.escaped:
                    continue
                monster_obs[num_monsters] = self._encode_monster(monster)
                num_monsters += 1

        # --- Entity counts (normalized) ---
        num_hand = min(len(player.hand), MAX_HAND_SIZE) if in_combat else 0
        entity_counts = np.array(
            [num_hand / MAX_HAND_SIZE, num_monsters / MAX_MONSTERS],
            dtype=np.float32,
        )

        return {
            "player": player_obs,
            "hand": hand_obs,
            "monsters": monster_obs,
            "entity_counts": entity_counts,
        }

    @staticmethod
    def _encode_card(card: dg.Card) -> np.ndarray:
        features = np.zeros(CARD_FEATURES, dtype=np.float32)
        features[0] = min(max(card.cost_for_turn, 0) / 5.0, 1.0)
        # One-hot card type (indices 1-5)
        one_hot = _card_type_one_hot(card.card_type)
        features[1:6] = one_hot
        # Damage
        if card.damage is not None:
            features[6] = min(card.damage / 50.0, 1.0)
        # Block
        if card.block is not None:
            features[7] = min(card.block / 50.0, 1.0)
        return features

    @staticmethod
    def _encode_monster(monster: dg.Monster) -> np.ndarray:
        features = np.zeros(MONSTER_FEATURES, dtype=np.float32)
        features[0] = monster.current_health / max(monster.max_health, 1)
        features[1] = min(monster.max_health / 100.0, 1.0)
        features[2] = min(monster.current_block / 50.0, 1.0)

        # Intent one-hot (indices 3-6)
        move = monster.next_move
        intent = move.get_intent() if move is not None else None
        features[3:7] = _intent_one_hot(intent)

        # Move damage
        if move is not None:
            dmg = move.get_damage()
            if dmg is not None:
                features[7] = min(dmg / 50.0, 1.0)

        return features

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_reward(self, raw_reward: float, is_terminal: bool, info: dict) -> float:
        reward = -0.001  # step cost

        # Terminal outcomes
        if is_terminal:
            if info.get("win"):
                return 1.0
            else:
                return -0.2

        # Illegal action penalty (use game's own penalty)
        if info.get("illegal"):
            return -0.01

        player = self._game.ctx.player
        dungeon = self._game.ctx.d

        # Floor progress
        if dungeon.floor_num > self._prev_floor:
            reward += 0.15

        # Combat damage dealt (dense signal)
        current_ratio = self._get_total_monster_hp_ratio()
        if current_ratio < self._prev_monster_hp_ratio:
            reward += 0.02 * (self._prev_monster_hp_ratio - current_ratio)

        # Player took damage (penalty)
        if player.current_health < self._prev_player_hp:
            hp_lost_ratio = (self._prev_player_hp - player.current_health) / max(player.max_health, 1)
            reward -= 0.05 * hp_lost_ratio

        return reward

    def _get_total_monster_hp_ratio(self) -> float:
        """Sum of (current_hp / max_hp) across all alive monsters in combat."""
        try:
            room = self._game.ctx.d.get_curr_room()
            if room.phase != dg.RoomPhase.COMBAT or room.monster_group is None:
                return 0.0
            total = 0.0
            for m in room.monster_group:
                if not m.is_dead and not m.escaped:
                    total += m.current_health / max(m.max_health, 1)
            return total
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "steps": self._steps,
            "floor": self._game.ctx.d.floor_num,
        }
        try:
            room = self._game.ctx.d.get_curr_room()
            info["in_combat"] = room.phase == dg.RoomPhase.COMBAT
            info["player_hp"] = self._game.ctx.player.current_health
        except Exception:
            pass
        return info
