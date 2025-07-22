# In air_defense_ai/models/radar_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network for radar signal classification.
    """
    def __init__(self, num_classes=24):
        super(RadarCNN, self).__init__()
        
        # Input shape: (batch_size, 2, 1024) where 2 is for I and Q channels
        
        # Convolutional Block 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 1024 -> 512
        
        # Convolutional Block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 512 -> 256
        
        # Convolutional Block 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2) # 256 -> 128
        
        # Classifier Head
        self.flatten = nn.Flatten()
        # The flattened size is out_channels * (sequence_length / 8) = 256 * 128
        self.fc1 = nn.Linear(256 * 128, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Classifier
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # Logits output
        
        return x