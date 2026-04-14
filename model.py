import torch.nn as nn
from modules import ConvBlock, EncoderBlock, DecoderBlock, AttentionGate

class AttentionUNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        
        # Initial Channel Count = 8 (as per notes)
        base = 8

        # Encoder: 4 layers of 1D convolutions
        self.down1 = EncoderBlock(n_channels, base, kernel_size=15)   # 400 -> 200
        self.down2 = EncoderBlock(base, base*2, kernel_size=11)       # 200 -> 100
        self.down3 = EncoderBlock(base*2, base*4, kernel_size=7)      # 100 -> 50
        self.down4 = EncoderBlock(base*4, base*8, kernel_size=5)      # 50 -> 25 (Bottleneck)

        # Bottlnech(25 points)
        self.bottleneck = ConvBlock(base*8, base*16, kernel_size=3)

        # Attention gates
        # F_g: gate signal, F_l: skip signal, F_int: intermediate
        self.att4 = AttentionGate(F_g=base*8, F_l=base*8, F_int=base*4)
        self.att3 = AttentionGate(F_g=base*4, F_l=base*4, F_int=base*2)
        self.att2 = AttentionGate(F_g=base*2, F_l=base*2, F_int=base)
        self.att1 = AttentionGate(F_g=base,   F_l=base,   F_int=base//2)

        # Decoder: 1D Transposed Convolutions
        self.up4 = DecoderBlock(base*16, base*8, kernel_size=5)
        self.up3 = DecoderBlock(base*8,  base*4, kernel_size=7)
        self.up2 = DecoderBlock(base*4,  base*2, kernel_size=11)
        self.up1 = DecoderBlock(base*2,  base,   kernel_size=15)

        # Output Layer
        self.outc = nn.Conv1d(base, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        # Bottleneck (25 points)
        x = self.bottleneck(x)

        # Decoder with Skip Connections and Attention
        x = self.up4(x, s4, self.att4)
        x = self.up3(x, s3, self.att3)
        x = self.up2(x, s2, self.att2)
        x = self.up1(x, s1, self.att1)

        return self.outc(x)