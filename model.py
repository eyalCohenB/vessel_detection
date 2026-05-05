# models.py
import torch
import torch.nn as nn

class VesselNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d((2, 1)),
        )

        self.column_head = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1)
        )

        self.presence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        f = self.features(x)  # (B, 64, H', W)

        # presence
        presence_feat = f.mean(dim=(2, 3))
        presence_logit = self.presence_head(presence_feat).squeeze(1)

        # column
        col_feat = f.mean(dim=2)
        column_logits = self.column_head(col_feat).squeeze(1)

        return presence_logit, column_logits