"""Card encoder and autoencoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from encoder.featurizer import INPUT_DIM

HIDDEN_DIM = 64
EMBED_DIM = 32


class CardEncoder(nn.Module):
    """MLP that maps a featurized card vector to a fixed-size embedding."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            # No final activation — raw embedding space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CardAutoencoder(nn.Module):
    """Autoencoder for pre-training the card encoder via reconstruction."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.encoder = CardEncoder(input_dim, hidden_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction
