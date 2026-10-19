import torch
from model import AttentionUNet1D


def test_output_has_same_shape_as_input():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    fake_signal = torch.randn(1, 1, 400)  # batch=1, channel=1, 400 samples

    with torch.no_grad():
        output = model(fake_signal)

    assert output.shape == fake_signal.shape
