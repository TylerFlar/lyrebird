"""Pre-train the card autoencoder on card_corpus.json.

Usage:
    python -m scripts.pretrain_encoder [--epochs 1000] [--embed-dim 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encoder.card_encoder import CardAutoencoder
from encoder.featurizer import INPUT_DIM, featurize_card, featurize_corpus


def train_autoencoder(
    corpus: list[dict],
    epochs: int = 1000,
    lr: float = 1e-3,
    embed_dim: int = 32,
    filter_game: str | None = None,
) -> tuple[CardAutoencoder, np.ndarray]:
    """Train the autoencoder. Returns (model, feature_matrix)."""
    if filter_game:
        corpus = [c for c in corpus if c["game"] == filter_game]

    features = featurize_corpus(corpus)
    X = torch.tensor(features)

    model = CardAutoencoder(INPUT_DIM, hidden_dim=64, embed_dim=embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, reconstruction = model(X)
        loss = loss_fn(reconstruction, X)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.6f}")

    return model, features


def print_nearest_neighbors(
    corpus: list[dict], features: np.ndarray, encoder: nn.Module, n: int = 3
):
    """Print nearest-neighbor cards for a few probe cards."""
    X = torch.tensor(features)
    with torch.no_grad():
        embeddings = encoder(X).numpy()

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    sim = normed @ normed.T

    # Probes: pick a few StS cards and show nearest Balatro neighbors
    sts_indices = [i for i, c in enumerate(corpus) if c["game"] == "sts"]
    probes = sts_indices[:5] if len(sts_indices) >= 5 else sts_indices

    print("\n-- Nearest Neighbors --")
    for idx in probes:
        card = corpus[idx]
        sims = sim[idx]
        # Exclude self, rank by similarity
        ranked = np.argsort(-sims)
        print(f"\n  {card['name']} ({card['game']}, {card['card_type']}):")
        shown = 0
        for r in ranked:
            if r == idx:
                continue
            neighbor = corpus[r]
            print(f"    {sims[r]:.3f}  {neighbor['name']} ({neighbor['game']}, {neighbor['card_type']})")
            shown += 1
            if shown >= n:
                break


def save_tsne_plot(
    corpus: list[dict], features: np.ndarray, encoder: nn.Module, path: Path
):
    """Save a t-SNE visualization of card embeddings."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Skipping t-SNE plot (matplotlib/sklearn not installed)")
        return

    X = torch.tensor(features)
    with torch.no_grad():
        embeddings = encoder(X).numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(corpus) - 1))
    coords = tsne.fit_transform(embeddings)

    games = [c["game"] for c in corpus]
    card_types = [c["card_type"] for c in corpus]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by game
    ax = axes[0]
    for game in sorted(set(games)):
        mask = [g == game for g in games]
        ax.scatter(coords[mask, 0], coords[mask, 1], label=game, alpha=0.6, s=20)
    ax.set_title("Colored by game")
    ax.legend()

    # Color by card_type
    ax = axes[1]
    for ct in sorted(set(card_types)):
        mask = [t == ct for t in card_types]
        ax.scatter(coords[mask, 0], coords[mask, 1], label=ct, alpha=0.6, s=20)
    ax.set_title("Colored by card_type")
    ax.legend(fontsize=7)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  t-SNE plot saved to {path}")


def reconstruction_loss_by_game(
    corpus: list[dict], features: np.ndarray, model: CardAutoencoder
):
    """Print reconstruction loss broken down by game."""
    model.eval()
    loss_fn = nn.MSELoss(reduction="none")

    X = torch.tensor(features)
    with torch.no_grad():
        _, reconstruction = model(X)
    per_sample = loss_fn(reconstruction, X).mean(dim=1).numpy()

    print("\n-- Reconstruction Loss by Game --")
    for game in ["balatro", "sts"]:
        mask = [c["game"] == game for c in corpus]
        if any(mask):
            losses = per_sample[mask]
            print(f"  {game:10s}: mean={losses.mean():.6f}  std={losses.std():.6f}  n={losses.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train card encoder")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    corpus_path = _ROOT / "data" / "card_corpus.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} cards from {corpus_path}")

    # Train on full corpus
    print("\nTraining autoencoder on full corpus...")
    model, features = train_autoencoder(corpus, epochs=args.epochs, lr=args.lr, embed_dim=args.embed_dim)

    # Save encoder weights
    models_dir = _ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    encoder_path = models_dir / "card_encoder.pt"
    torch.save(model.encoder.state_dict(), str(encoder_path))
    print(f"\nEncoder saved to {encoder_path}")

    # Also train and save StS-only encoder
    print("\nTraining autoencoder on StS cards only...")
    model_sts, features_sts = train_autoencoder(
        corpus, epochs=args.epochs, lr=args.lr, embed_dim=args.embed_dim, filter_game="sts"
    )
    sts_encoder_path = models_dir / "card_encoder_sts_only.pt"
    torch.save(model_sts.encoder.state_dict(), str(sts_encoder_path))
    print(f"StS-only encoder saved to {sts_encoder_path}")

    # Analysis on full model
    print_nearest_neighbors(corpus, features, model.encoder)
    reconstruction_loss_by_game(corpus, features, model)
    save_tsne_plot(corpus, features, model.encoder, _ROOT / "data" / "embedding_tsne.png")


if __name__ == "__main__":
    main()
