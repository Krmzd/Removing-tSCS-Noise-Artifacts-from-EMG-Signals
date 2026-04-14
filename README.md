# Attention-U-Net-for-EMG-Denoising
Removing tSCS Artifacts to Clarify Cutaneous Reflexes

## Project Overview
This project implements a deep learning solution to a problem in neurophysiology. During transcutaneous spinal cord stimulation (tSCS), high amplitude artifacts obscure the biological signals of interest. This project uses Attention U-Net to extract tSCS artifacts while preserving the biological signals.

## Main Tasks
- **Noise Extraction:** Frequency-locked template extraction and adaptive statistical thresholding to isolate stimulation noise.
- **Synthetic Training Pipeline:** Implementation of supervised learning strategy where pure artifacts are added into clean EMG to create perfectly synchronized training pairs.
- **Attention U-Net Architecture:** A 1D convolutional neural network with Attention Gates on skip connections to prioritize artifacts and protect the biological signal.

## Project Structure
```text
├── models/
│   ├── attention_unet.py      # Attention U-Net 1D Architecture
│   └── modules.py             # Encoder, Decoder, and Attention Gate blocks
├── utils/
│   ├── data_loader.py         # Adaptive thresholding and synthetic mixing logic
│   ├── process.py             # EMG rectification and reflex segmentation
│   └── visualization.py       # Learning curves and signal comparison plots
├── train.py                   # Main training script (Scaled Huber Loss)
├── test_denoiser.py           # Real-world inference and CSV generation
├── validation_denoiser.py     # Accuracy validation against Ground Truth
└── unet_denoiser.pth          # Saved model weights (Checkpoint)
```
## Dataset and Preprocessing 
- **Signal Type:** Multichannel EMG(8 channels)
- **Synthetic Strategy:** Because noisy and clean signals are asynchronous, I implemented a frequency-locked template extraction to isolate "pure" noise, which was then added to clean EMG to creat synchronized training pairs.
- **Normalization:** Global Max-Abs scaling to maintain the amplitude ratio between signal and noise.

