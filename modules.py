import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """1D Conv -> BatchNormalization -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """Focuses on artifacts while preserving the 0.02mV reflex"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class EncoderBlock(nn.Module):
    """Conv -> BN -> LeakyReLU then MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled # Return both for skip connection and next layer

class DecoderBlock(nn.Module):
    """Transposed Conv -> Attention -> Concat -> ConvBlock"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 1D Transposed Convolution (doubles the length)
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, kernel_size)

    def forward(self, x, skip, att_gate):
        x = self.up(x)
        # Apply Attention to the skip connection (from encoder)
        skip = att_gate(g=x, x=skip)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)