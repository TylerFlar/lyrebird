"""Run the cross-game transfer experiment.

Conditions:
  baseline           — raw 8-dim card features (no encoder)
  sts_only_frozen    — encoder pre-trained on StS cards only, frozen
  cross_game_frozen  — encoder pre-trained on full corpus, frozen
  cross_game_finetune — encoder pre-trained on full corpus, fine-tuned with PPO

Each condition is run 3× with different seeds.

Usage:
    python -m scripts.run_transfer_experiment [--timesteps 500000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encoder.encoder_wrapper import EncoderObsWrapper
from envs.sts_gymnasium_wrapper import SlayTheSpireEnv
from scripts.train_sts_ppo import STSLoggingCallback


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(
    encoder_path: str | Path | None = None,
    freeze_encoder: bool = True,
) -> ActionMasker:
    """Create the StS env, optionally wrapped with the encoder."""
    env = SlayTheSpireEnv(max_hp=80, energy_per_turn=3, max_steps=2000)
    if encoder_path is not None:
        env = EncoderObsWrapper(
            env,
            encoder_path=encoder_path,
            game="sts",
            freeze_encoder=freeze_encoder,
        )
    return ActionMasker(env, mask_fn)


def run_condition(
    name: str,
    run_dir: Path,
    timesteps: int,
    seed: int,
    encoder_path: str | Path | None = None,
    freeze_encoder: bool = True,
):
    """Train one condition with one seed."""
    tag = f"{name}/seed_{seed}"
    log_dir = run_dir / name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Condition: {name}  seed={seed}  timesteps={timesteps}")
    print(f"  Encoder: {encoder_path or 'None'}  freeze={freeze_encoder}")
    print(f"{'='*60}")

    env = make_env(encoder_path=encoder_path, freeze_encoder=freeze_encoder)

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
        verbose=0,
        tensorboard_log=str(log_dir),
        seed=seed,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=STSLoggingCallback(),
        tb_log_name=f"seed_{seed}",
    )
    elapsed = time.time() - t0

    model_path = log_dir / f"model_seed_{seed}"
    model.save(str(model_path))
    print(f"  Done in {elapsed:.1f}s. Model saved to {model_path}")
    env.close()


CONDITIONS = {
    "baseline": {
        "encoder_path": None,
        "freeze_encoder": True,
    },
    "sts_only_frozen": {
        "encoder_path": _ROOT / "models" / "card_encoder_sts_only.pt",
        "freeze_encoder": True,
    },
    "cross_game_frozen": {
        "encoder_path": _ROOT / "models" / "card_encoder.pt",
        "freeze_encoder": True,
    },
    "cross_game_finetune": {
        "encoder_path": _ROOT / "models" / "card_encoder.pt",
        "freeze_encoder": False,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run transfer experiment")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per condition")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Run only these conditions (default: all)")
    args = parser.parse_args()

    run_dir = _ROOT / "runs" / "transfer"
    run_dir.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions or list(CONDITIONS.keys())
    seed_list = [42 + i * 17 for i in range(args.seeds)]

    # Verify encoder weights exist for non-baseline conditions
    for cond_name in conditions:
        cfg = CONDITIONS[cond_name]
        if cfg["encoder_path"] is not None and not Path(cfg["encoder_path"]).exists():
            print(f"ERROR: Encoder weights not found: {cfg['encoder_path']}")
            print("Run `python -m scripts.pretrain_encoder` first.")
            sys.exit(1)

    total_runs = len(conditions) * len(seed_list)
    run_idx = 0

    for cond_name in conditions:
        cfg = CONDITIONS[cond_name]
        for seed in seed_list:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}]", end="")
            run_condition(
                name=cond_name,
                run_dir=run_dir,
                timesteps=args.timesteps,
                seed=seed,
                encoder_path=cfg["encoder_path"],
                freeze_encoder=cfg["freeze_encoder"],
            )

    print(f"\n\nAll {total_runs} runs complete. Logs in {run_dir}")
    print("Run `python -m scripts.plot_transfer_results` to generate figures.")


if __name__ == "__main__":
    main()
