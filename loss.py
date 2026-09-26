import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks

# Huber Loss
def get_huber_loss(beta=0.1):
    return nn.SmoothL1Loss(beta=beta)

# Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # log(cosh(x)) can be simplified for numerical stability:
        # log((exp(x) + exp(-x))/2)
        err = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(err + 1e-12)))

# Weighted Loss 
class ScaledLoss(nn.Module):
    def __init__(self, base_loss_fn, scale=100.0):
        super().__init__()
        self.loss_fn = base_loss_fn
        self.scale = scale

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true) * self.scale


# ================= Evaluation metrics =================

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def snr_db(clean, estimate):
    return float(10 * np.log10(np.sum(clean ** 2) / np.sum((estimate - clean) ** 2)))


# Part 1: synthetic (true clean signal is known)
def synthetic_metrics(clean, noisy, pred):
    mask = np.abs(noisy - clean) > 1e-8   # samples where a spike was added
    return {
        "RMSE_input": rmse(clean, noisy),
        "RMSE_output": rmse(clean, pred),
        "Corr_output": corr(clean, pred),
        "SNR_improvement_dB": snr_db(clean, pred) - snr_db(clean, noisy),
        "RMSE_artifact_before": rmse(clean[mask], noisy[mask]),
        "RMSE_artifact_after": rmse(clean[mask], pred[mask]),
        "RMSE_clean_parts": rmse(clean[~mask], pred[~mask]),   # ideal = 0
    }


# Part 2: real tSCS (no true clean signal)
def real_metrics(raw, cleaned, h_mult, pre_w, post_w, tail_len=20000, edge=400, distance=40):
    raw = raw - np.mean(raw)                        # same centering as the model input
    raw_c, clean_c = raw[:-edge], cleaned[:-edge]   # last samples are never cleaned

    # Find tSCS spikes the same way as in preprocessing
    thresh = np.std(raw_c[-tail_len:]) * h_mult
    peaks_before, _ = find_peaks(np.abs(raw_c), height=thresh, distance=distance)
    peaks_after, _ = find_peaks(np.abs(clean_c), height=thresh, distance=distance)

    # How much smaller is each spike after cleaning?
    ratios = []
    for p in peaks_before:
        s, e = max(0, p - pre_w), min(len(raw_c), p + post_w)
        before = np.max(np.abs(raw_c[s:e]))
        if before > 0:
            ratios.append(np.max(np.abs(clean_c[s:e])) / before)
    ratio = float(np.median(ratios)) if ratios else float("nan")

    # Clean tail: raw is already clean here, so cleaned should match it
    return {
        "Spikes_before": len(peaks_before),
        "Spikes_after": len(peaks_after),
        "Peak_reduction_dB": 20 * np.log10(ratio) if ratio > 0 else float("nan"),
        "Tail_RMSE": rmse(raw_c[-tail_len:], clean_c[-tail_len:]),
        "Tail_Corr": corr(raw_c[-tail_len:], clean_c[-tail_len:]),
    }
