import torch
from model import AttentionUNet1D


def test_output_has_same_shape_as_input():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()

    fake_signal = torch.randn(1, 1, 400)  # batch=1, channel=1, 400 samples

    with torch.no_grad():
        output = model(fake_signal)

    assert output.shape == fake_signal.shape


def test_output_has_no_nan():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()
    fake_signal = torch.randn(1, 1, 400)
    with torch.no_grad():
        output = model(fake_signal)
    assert not torch.isnan(output).any()


def test_works_with_a_batch():
    model = AttentionUNet1D(n_channels=1, n_classes=1)
    model.eval()
    fake_batch = torch.randn(8, 1, 400)  # 8 signals at once, like in training
    with torch.no_grad():
        output = model(fake_batch)
    assert output.shape == (8, 1, 400)
