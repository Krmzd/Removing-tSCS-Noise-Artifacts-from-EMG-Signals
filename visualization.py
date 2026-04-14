import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.tools as tls
import plotly.io as pio

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

    def plot_signal_comparison(self, input_sig, target_sig, pred_sig, window_idx=0):
        """2. Overlays Input, Target, and Prediction to see denoising quality"""
        # Ensure data is numpy
        if torch.is_tensor(input_sig): input_sig = input_sig.detach().cpu().numpy().flatten()
        if torch.is_tensor(target_sig): target_sig = target_sig.detach().cpu().numpy().flatten()
        if torch.is_tensor(pred_sig): pred_sig = pred_sig.detach().cpu().numpy().flatten()

        plt.figure(figsize=(15, 6))
        plt.plot(input_sig, label='Input (Noisy)', color='lightskyblue', alpha=0.6)
        plt.plot(target_sig, label='Target (Ground Truth)', color='forestgreen', linewidth=1.5)
        plt.plot(pred_sig, label='U-Net Prediction', color='crimson', linestyle='--', linewidth=1.5)
        
        plt.title(f"Signal Comparison - Window {window_idx}")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)
        plt.show()

    def plot_residual(self, target_sig, pred_sig):
        """3. Plots the ERROR (what the model missed)"""
        if torch.is_tensor(target_sig): target_sig = target_sig.detach().cpu().numpy().flatten()
        if torch.is_tensor(pred_sig): pred_sig = pred_sig.detach().cpu().numpy().flatten()
        
        residual = target_sig - pred_sig
        
        plt.figure(figsize=(15, 4))
        plt.plot(residual, color='purple', linewidth=1)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title("Residual Plot (Target - Prediction)\nShould look like random noise if successful")
        plt.xlabel("Samples")
        plt.ylabel("Error Magnitude")
        plt.grid(True, alpha=0.2)
        plt.show()

    def plot_average(self, first_data, second_data, x_label, y_label, figure_name, color_one= 'blue', color_two="black"):
        
        min_data_one= np.nanmin(first_data)
        min_data_two= np.nanmin(second_data)
        min_data = np.min([min_data_one, min_data_two])
        
        max_data_one= np.nanmax(first_data)
        max_data_two= np.nanmax(second_data)
        max_data = np.max([max_data_one, max_data_two])

        range_data= np.abs(max_data - min_data)
        x_time = np.arange(-50, 200, 0.5)
        x_lim= (-50, np.max(x_time))
        # x_lim= (0, len(first_data))
        y_lim= (0, 0.5)
        fig = plt.figure()
        plt.plot(x_time, first_data, color_one, label="Pre-BLT")
        plt.plot(x_time, second_data, color_two, label="Post-BLT")
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        plotly_fig = tls.mpl_to_plotly(fig)
        plotly_fig.update_layout(width=800, height= 600)
        pio.write_html(plotly_fig, f'{self.save_directory}/{figure_name}.html')
    
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

    def plot_error_distribution(self, target, pred):
        # Shows if errors are random or systematic
        error = target - pred
        plt.figure(figsize=(8, 5))
        plt.hist(error, bins=100, color='purple', alpha=0.7)
        plt.axvline(0, color='black', linestyle='--')
        plt.title("Error Distribution (Residuals)")
        plt.xlabel("Error Magnitude (mV)")
        plt.ylabel("Frequency")
        plt.show()
