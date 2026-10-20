import numpy as np
import torch

from config import Device, Model_path
from model import AttentionUNet1D

WINDOW_SIZE = 400
STEP_SIZE = 200  # 50% overlap


def load_model(model_path=Model_path, device=Device):
    """Load the trained U-Net once and put it in evaluation mode."""
    model = AttentionUNet1D(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def denoise_signal(model, raw_signal, device=Device):
    """Clean one EMG signal (list or numpy array). Same steps as test_denoiser.py."""
    raw_signal = np.nan_to_num(np.asarray(raw_signal, dtype=np.float64))

    # Zero-centering and max-abs scaling
    centered = raw_signal - np.mean(raw_signal)
    scale_factor = np.max(np.abs(centered))
    if scale_factor == 0:
        return centered  # flat signal: nothing to clean
    normalized = centered / scale_factor

    output_buffer = np.zeros_like(normalized)
    count_mask = np.zeros_like(normalized)

    starts = list(range(0, len(normalized) - WINDOW_SIZE + 1, STEP_SIZE))
    if starts[-1] != len(normalized) - WINDOW_SIZE:
        starts.append(len(normalized) - WINDOW_SIZE)

    with torch.no_grad():
        for start in starts:
            end = start + WINDOW_SIZE
            window = torch.tensor(normalized[start:end]).float().unsqueeze(0).unsqueeze(0).to(device)
            cleaned_window = model(window).cpu().squeeze().numpy()
            output_buffer[start:end] += cleaned_window
            count_mask[start:end] += 1

    cleaned = output_buffer / np.maximum(count_mask, 1)
    return cleaned * scale_factor
