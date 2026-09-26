import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from visualization import EMGVisualizer
from model import AttentionUNet1D
from data_loader import get_loader
from loss import ScaledLoss, get_huber_loss
from config import Device, Processed_data_path, Model_path

device = Device
print("Using device:", device)
epochs = 100
batch_size = 32
learning_rate = 1e-4
reflex_sensitivity = 500.0  
viz = EMGVisualizer()

def evaluate(model, loader, criterion):
    """Average loss over a loader, in eval mode (no training, no gradients)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for synthetic_noisy, clean_target in loader:
            synthetic_noisy = synthetic_noisy.to(device)
            clean_target = clean_target.to(device)
            total += criterion(model(synthetic_noisy), clean_target).item()
    return total / len(loader)


def train_denoiser():
    # noisy_input: (Clean Recording + Added Spikes)
    # clean_target: (Original Clean Recording)
    train_parts = ["AA", "DA", "KM", "MJ", "MT", "NM", "Reihane", "SA", "SH", "Shubhman", "TS", "VI", "VIm"]  # YK held out as unseen test participant
    val_parts = ["NS", "MY"] 
    train_loader, val_loader = get_loader(Processed_data_path, train_parts, val_parts, batch_size=batch_size)

    model = AttentionUNet1D(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss:
    huber = get_huber_loss(beta=0.1)
    criterion = ScaledLoss(base_loss_fn=huber, scale=reflex_sensitivity)

    train_losses, val_losses = [], []

    # Early stopping
    patience = 10          # stop after 10 epochs with no improvement
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        
        loop = tqdm(train_loader, leave=True)
        for synthetic_noisy, clean_target in loop:
            synthetic_noisy = synthetic_noisy.to(device)
            clean_target = clean_target.to(device)

            # U-Net remove the added spikes and output the clean signal
            prediction = model(synthetic_noisy)
            
            # Compare the prediction to the ground-truth 
            loss = criterion(prediction, clean_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        # Measure train and validation loss the same way: end of epoch, eval mode
        avg_train_loss = evaluate(model, train_loader, criterion)
        avg_val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

      
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    viz_learning_curve = EMGVisualizer()
    viz_learning_curve.plot_learning_curve(train_losses, val_losses)

if __name__ == "__main__":
    train_denoiser()