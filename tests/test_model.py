"""Tests for the Attention U-Net model (model.py).

These tests use random fake signals, so they need no real data and no trained weights.

"""
import torch
from model import AttentionUNet1D


def test_output_has_same_shape_as_input():
    """A denoiser must return a signal of exactly the same shape it receives."""
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()  # inference mode, no training

    fake_signal = torch.randn(1, 1, 400)  # batch=1, channel=1, 400 samples (one window)

    with torch.no_grad():  # no gradients needed when only predicting
        output = model(fake_signal)

    assert output.shape == fake_signal.shape


def test_output_has_no_nan():
    """The model output must contain numbers, no NaN values."""
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    fake_signal = torch.randn(1, 1, 400)

    with torch.no_grad():
        output = model(fake_signal)

    assert not torch.isnan(output).any()


def test_works_with_a_batch():
    """The model must handle several signals at once, like during training."""
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    fake_batch = torch.randn(8, 1, 400)  # 8 signals at once

    with torch.no_grad():
        output = model(fake_batch)

    assert output.shape == (8, 1, 400)
