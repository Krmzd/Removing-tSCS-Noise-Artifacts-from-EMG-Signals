"""Tests for the cleaning function denoise_signal (inference.py).

An untrained model is used, so no trained weights or real data are needed.
These tests check the sliding-window logic, not the quality of the denoising.
"""
import numpy as np
from model import AttentionUNet1D
from inference import denoise_signal


def test_denoise_returns_same_length():
    """The cleaned signal must have the same length as the input signal,
    even when the length is not a multiple of the window size (400)."""
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    signal = np.random.randn(1000)  # 1000 samples: 400-sample windows don't fit evenly

    cleaned = denoise_signal(model, signal, device="cpu")

    assert len(cleaned) == len(signal)


def test_flat_signal_does_not_crash():
    """A completely flat signal (all zeros) must not crash the function.
    Without special handling, scaling would divide by zero."""
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    flat = np.zeros(800)  # all zeros, so the scale factor is 0

    cleaned = denoise_signal(model, flat, device="cpu")

    assert len(cleaned) == 800
