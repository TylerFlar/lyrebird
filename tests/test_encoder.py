"""Tests for encoder.card_encoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from encoder.card_encoder import EMBED_DIM, HIDDEN_DIM, CardAutoencoder, CardEncoder
from encoder.featurizer import INPUT_DIM


class TestCardEncoder:
    def test_output_shape(self):
        encoder = CardEncoder()
        x = torch.randn(5, INPUT_DIM)
        out = encoder(x)
        assert out.shape == (5, EMBED_DIM)

    def test_single_input(self):
        encoder = CardEncoder()
        x = torch.randn(1, INPUT_DIM)
        out = encoder(x)
        assert out.shape == (1, EMBED_DIM)

    def test_custom_dims(self):
        encoder = CardEncoder(input_dim=50, hidden_dim=32, embed_dim=16)
        x = torch.randn(3, 50)
        out = encoder(x)
        assert out.shape == (3, 16)

    def test_deterministic(self):
        encoder = CardEncoder()
        encoder.eval()
        x = torch.randn(3, INPUT_DIM)
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)
        assert torch.allclose(out1, out2)


class TestCardAutoencoder:
    def test_output_shapes(self):
        model = CardAutoencoder()
        x = torch.randn(5, INPUT_DIM)
        embedding, reconstruction = model(x)
        assert embedding.shape == (5, EMBED_DIM)
        assert reconstruction.shape == (5, INPUT_DIM)

    def test_reconstruction_loss_decreases(self):
        """Quick sanity check that a few training steps reduce loss."""
        model = CardAutoencoder()
        x = torch.randn(20, INPUT_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        model.train()
        _, recon = model(x)
        loss_before = loss_fn(recon, x).item()

        for _ in range(50):
            optimizer.zero_grad()
            _, recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()

        _, recon = model(x)
        loss_after = loss_fn(recon, x).item()
        assert loss_after < loss_before


class TestSaveLoad:
    def test_encoder_save_load(self):
        encoder = CardEncoder()
        x = torch.randn(3, INPUT_DIM)

        with torch.no_grad():
            out_before = encoder(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            torch.save(encoder.state_dict(), str(path))

            loaded = CardEncoder()
            loaded.load_state_dict(torch.load(str(path), weights_only=True))
            loaded.eval()

            with torch.no_grad():
                out_after = loaded(x)

        assert torch.allclose(out_before, out_after)
