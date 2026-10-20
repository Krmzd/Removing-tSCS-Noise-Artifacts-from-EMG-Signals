import numpy as np
from model import AttentionUNet1D
from inference import denoise_signal


def test_denoise_returns_same_length():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()
    signal = np.random.randn(1000)   # 1000 samples, not a multiple of 400
    cleaned = denoise_signal(model, signal, device="cpu")
    assert len(cleaned) == len(signal)


def test_flat_signal_does_not_crash():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()
    flat = np.zeros(800)             # all zeros: scale_factor would be 0
    cleaned = denoise_signal(model, flat, device="cpu")
    assert len(cleaned) == 800
