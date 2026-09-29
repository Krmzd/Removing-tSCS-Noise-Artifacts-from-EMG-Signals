import matplotlib.pyplot as plt
import numpy as np
import torch

class EMGVisualizer:
    def __init__(self, fs=2000):
        self.fs = fs  # Sampling rate for time-axis calculations

    def plot_learning_curve(self, train_losses, val_losses, save_path=None):
        """1. Plots the Train vs Validation Loss over Epochs"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
        plt.title("Learning Curve: Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Scaled Huber)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_residual(self, raw, cleaned):
        """3. Plots what the model REMOVED from the signal (Raw - Cleaned)"""
        if torch.is_tensor(raw): raw = raw.detach().cpu().numpy().flatten()
        if torch.is_tensor(cleaned): cleaned = cleaned.detach().cpu().numpy().flatten()

        removed = (raw - np.mean(raw)) - cleaned   # same centering as the model input

        plt.figure(figsize=(15, 4))
        plt.plot(removed, color='purple', linewidth=1)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title("Removed by the Model (Raw - Cleaned)\nShould contain only the tSCS spikes, not muscle activity")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.2)
        plt.show()
    
    def plot_inference_check(self, raw, cleaned, muscle_name, fs=2000):

        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Full Signal
        plt.subplot(2, 1, 1)
        plt.plot(raw, color='deepskyblue', alpha=0.5, label='Raw Noisy Signal')
        plt.plot(cleaned, color='black', label='U-Net Cleaned', linewidth=0.7)
        plt.title(f"Cleaning Result: {muscle_name}")
        plt.legend()

        # Subplot 2: The Last 10,000 samples
        plt.subplot(2, 1, 2)
        plt.plot(raw[-10000:], color='deepskyblue', alpha=0.5, label='Original Tail')
        plt.plot(cleaned[-10000:], color='black', label='Cleaned Tail')
        plt.title("Validation Check: Last 10,000 Samples")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
