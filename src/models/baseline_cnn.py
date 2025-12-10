import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR100BaselineCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        # --- Feature extractor (convolutional part) ---
        self.features = nn.Sequential(
            # Block 1: input [B, 3, 32, 32]
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 32, 16, 16]

            # Block 2: [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # -> [B, 64, 8, 8]

            # Block 3: [B, 64, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)               # -> [B, 128, 4, 4]
        )

        # --- Classifier (fully-connected part) ---
        # After the convs, the tensor is [B, 128, 4, 4] -> 128*4*4 = 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.features(x)        # -> [B, 128, 4, 4]
        x = torch.flatten(x, 1)     # -> [B, 2048]
        x = self.classifier(x)      # -> [B, num_classes]
        return x
