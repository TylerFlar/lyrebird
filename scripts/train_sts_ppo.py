"""Train MaskablePPO on Slay the Spire (Exordium / Silent).

Usage:
    python -m scripts.train_sts_ppo [--timesteps 500000] [--run-name sts_ppo]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Ensure lyrebird root is on path so envs is importable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.sts_gymnasium_wrapper import SlayTheSpireEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Action mask function for ActionMasker wrapper
# ---------------------------------------------------------------------------

def mask_fn(env: SlayTheSpireEnv) -> np.ndarray:
    return env.action_masks()


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class STSLoggingCallback(BaseCallback):
    """Log STS-specific metrics to tensorboard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_floors: list[int] = []
        self._episode_wins: list[bool] = []
        self._episode_hp: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "floor_reached" in info:
                self._episode_floors.append(info["floor_reached"])
                self._episode_wins.append(info.get("win", False))
            if "player_hp" in info and info.get("win", False):
                self._episode_hp.append(info["player_hp"])

        # Log every 100 episodes
        if len(self._episode_floors) >= 100:
            self.logger.record("sts/mean_floor_reached", np.mean(self._episode_floors))
            self.logger.record("sts/win_rate", np.mean(self._episode_wins))
            if self._episode_hp:
                self.logger.record("sts/mean_win_hp", np.mean(self._episode_hp))
            self.logger.record("sts/episodes", self.num_timesteps)
            self._episode_floors.clear()
            self._episode_wins.clear()
            self._episode_hp.clear()

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_env() -> ActionMasker:
    env = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=2000)
    return ActionMasker(env, mask_fn)


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Slay the Spire")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name", type=str, default="sts_ppo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = _ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.05,
        learning_rate=1e-4,
        n_steps=4096,
        clip_range=0.15,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=str(run_dir),
        seed=args.seed,
    )

    print(f"Training for {args.timesteps} timesteps. Logs: {run_dir}")
    t0 = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=STSLoggingCallback(),
    )

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s. Saving model...")

    model.save(str(run_dir / "sts_ppo_final"))
    print(f"Model saved to {run_dir / 'sts_ppo_final'}")


if __name__ == "__main__":
    main()
