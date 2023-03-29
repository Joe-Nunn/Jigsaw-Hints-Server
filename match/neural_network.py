"""
Contains class NeuralNetwork which defines a neural network to predict if an image of a jigsaw piece is from the same
part of the puzzle as an image of a section of the box
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Neural network to predict probabilities of a jigsaw piece being from the same part of the puzzle as a section
    of the base image
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=10),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout()
        )

        self.fc1 = nn.Linear(128 * 12 * 12, 8000)
        self.fcOut1 = nn.Linear(8000, 4000)
        self.fcOut2 = nn.Linear(4000, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, pieces, base_sections):
        """
        Calculates prediction that jigsaw piece and the sample of the base are the same location
        """
        pieces = self.conv(pieces)
        pieces = torch.flatten(pieces, start_dim=1)
        pieces = self.fc1(pieces)
        pieces = self.dropout(pieces)
        pieces = self.sigmoid(pieces)

        base_sections = self.conv(base_sections)
        base_sections = torch.flatten(base_sections, start_dim=1)
        base_sections = self.fc1(base_sections)
        base_sections = self.dropout(base_sections)
        base_sections = self.sigmoid(base_sections)

        diff = torch.abs(pieces - base_sections)

        fc_out_result = self.fcOut1(diff)
        fc_out_result = self.fcOut2(fc_out_result)
        return self.sigmoid(fc_out_result)
