import torch
import torch.nn as nn

class UNetSeparator(nn.Module):
    def __init__(self):
        super(UNetSeparator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 4, kernel_size=3, padding=1),  # 4 stems: vocals, drums, bass, other
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Shape: [batch, 4, time]