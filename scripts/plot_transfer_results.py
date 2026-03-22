"""Generate figures and tables from transfer experiment tensorboard logs.

Outputs:
  results/learning_curves.png   — floor_reached over timesteps per condition
  results/summary.csv           — final metrics per condition
  results/embedding_analysis.png — t-SNE + nearest-neighbor table

Usage:
    python -m scripts.plot_transfer_results
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_tb_scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Read scalar events from tensorboard event files.

    Returns list of (step, value) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed.")
        sys.exit(1)

    points = []
    for run_dir in sorted(log_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                points.append((event.step, event.value))
    return sorted(points)


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_learning_curves(transfer_dir: Path, output_path: Path):
    """Plot floor_reached learning curves for all conditions."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping learning curves plot (matplotlib not installed)")
        return

    conditions = ["baseline", "sts_only_frozen", "cross_game_frozen", "cross_game_finetune"]
    colors = ["#888888", "#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, color in zip(conditions, colors):
        cond_dir = transfer_dir / cond
        if not cond_dir.exists():
            continue

        # Collect data from all seed runs
        all_curves = []
        for seed_dir in sorted(cond_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            ea_path = seed_dir
            points = _read_tb_scalars(cond_dir, "sts/mean_floor_reached")
            if points:
                all_curves.append(points)

        if not all_curves:
            # Try reading from individual seed subdirs
            for seed_dir in sorted(cond_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                points = _read_tb_scalars(seed_dir, "sts/mean_floor_reached")
                if points:
                    all_curves.append(points)

        if not all_curves:
            print(f"  No data found for condition: {cond}")
            continue

        # Interpolate all curves to common x-axis
        all_steps = sorted(set(s for curve in all_curves for s, _ in curve))
        if not all_steps:
            continue

        interpolated = []
        for curve in all_curves:
            steps_arr = np.array([s for s, _ in curve])
            vals_arr = np.array([v for _, v in curve])
            interp = np.interp(all_steps, steps_arr, vals_arr)
            interpolated.append(interp)

        interpolated = np.array(interpolated)
        mean = _smooth(interpolated.mean(axis=0))
        std = interpolated.std(axis=0)

        ax.plot(all_steps, mean, label=cond, color=color, linewidth=2)
        ax.fill_between(all_steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Floor Reached")
    ax.set_title("StS Transfer Experiment: Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to {output_path}")


def generate_summary_table(transfer_dir: Path, output_path: Path):
    """Generate summary CSV with final metrics per condition."""
    conditions = ["baseline", "sts_only_frozen", "cross_game_frozen", "cross_game_finetune"]

    rows = []
    for cond in conditions:
        cond_dir = transfer_dir / cond
        if not cond_dir.exists():
            continue

        floor_points = _read_tb_scalars(cond_dir, "sts/mean_floor_reached")
        win_points = _read_tb_scalars(cond_dir, "sts/win_rate")

        final_floor = floor_points[-1][1] if floor_points else float("nan")
        final_win = win_points[-1][1] if win_points else float("nan")

        # Steps to reach floor 8
        steps_to_8 = float("nan")
        for step, val in floor_points:
            if val >= 8.0:
                steps_to_8 = step
                break

        rows.append({
            "condition": cond,
            "final_mean_floor": f"{final_floor:.2f}",
            "steps_to_floor_8": int(steps_to_8) if not np.isnan(steps_to_8) else "never",
            "final_win_rate": f"{final_win:.4f}" if not np.isnan(final_win) else "n/a",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "final_mean_floor", "steps_to_floor_8", "final_win_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary table saved to {output_path}")
    for row in rows:
        print(f"  {row['condition']:25s}  floor={row['final_mean_floor']}  "
              f"steps_to_8={row['steps_to_floor_8']}  win={row['final_win_rate']}")


def plot_embedding_analysis(output_path: Path):
    """Generate t-SNE visualization and nearest-neighbor analysis."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import torch
    except ImportError:
        print("Skipping embedding analysis (matplotlib/sklearn/torch not installed)")
        return

    from encoder.card_encoder import CardEncoder
    from encoder.featurizer import INPUT_DIM, featurize_corpus

    corpus_path = _ROOT / "data" / "card_corpus.json"
    encoder_path = _ROOT / "models" / "card_encoder.pt"

    if not encoder_path.exists():
        print("Skipping embedding analysis (no pre-trained encoder found)")
        return

    with open(corpus_path) as f:
        corpus = json.load(f)

    features = featurize_corpus(corpus)
    encoder = CardEncoder(INPUT_DIM)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu", weights_only=True))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.tensor(features)).numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(corpus) - 1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    games = [c["game"] for c in corpus]
    card_types = [c["card_type"] for c in corpus]

    # Plot with game as color, card_type as marker
    game_colors = {"balatro": "#2196F3", "sts": "#FF5722"}
    for game in ["balatro", "sts"]:
        mask = np.array([g == game for g in games])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=game, c=game_colors[game], alpha=0.6, s=25,
        )

    # Annotate StS cards
    for i, c in enumerate(corpus):
        if c["game"] == "sts":
            ax.annotate(c["name"], (coords[i, 0], coords[i, 1]),
                        fontsize=5, alpha=0.7)

    ax.set_title("Card Embeddings (pre-trained encoder)")
    ax.legend()

    # Nearest-neighbor table as text
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    sim = (embeddings / norms) @ (embeddings / norms).T

    nn_lines = ["Nearest cross-game neighbors:", ""]
    sts_indices = [i for i, c in enumerate(corpus) if c["game"] == "sts"]
    for idx in sts_indices[:8]:
        card = corpus[idx]
        ranked = np.argsort(-sim[idx])
        neighbors = []
        for r in ranked:
            if corpus[r]["game"] != card["game"]:
                neighbors.append(f"{corpus[r]['name']} ({sim[idx][r]:.2f})")
                if len(neighbors) >= 3:
                    break
        nn_lines.append(f"{card['name']:20s} → {', '.join(neighbors)}")

    fig.text(0.02, 0.02, "\n".join(nn_lines), fontsize=7, family="monospace",
             verticalalignment="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Embedding analysis saved to {output_path}")


def main():
    transfer_dir = _ROOT / "runs" / "transfer"
    results_dir = _ROOT / "results"

    if transfer_dir.exists():
        plot_learning_curves(transfer_dir, results_dir / "learning_curves.png")
        generate_summary_table(transfer_dir, results_dir / "summary.csv")
    else:
        print(f"No transfer experiment logs found at {transfer_dir}")
        print("Run `python -m scripts.run_transfer_experiment` first.")

    plot_embedding_analysis(results_dir / "embedding_analysis.png")


if __name__ == "__main__":
    main()
