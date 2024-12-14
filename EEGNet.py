import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from glob import glob
import pandas as pd

class EEGNet(nn.Module):
    def __init__(self, num_classes = 3):
        super(EEGNet, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 4), stride=1, padding=(0, 2))
        self.batch_norm1 = nn.BatchNorm2d(2)
        self.activation1 = nn.ReLU()

        # Ajouter la couche depthwise
        self.depthwise_conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 1), stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(4)
        self.activation2 = nn.ReLU()

        # Définir une couche conv2 (si nécessaire)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=1, padding=(0, 1))  # Par exemple
        self.batch_norm3 = nn.BatchNorm2d(8)
        self.activation3 = nn.ReLU()

        # Couches fully connected
        self.fc = nn.Linear(8 * 255 * 2, 64)  # 8 * 255 * 2 = 4080
        self.out = nn.Linear(64, num_classes)  # Par exemple, 28 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Appliquer les convolutions et activations
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.depthwise_conv1(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        x = self.conv2(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)


        # Appliquer la couche fully connected
        x = x.view(x.size(0), -1)  # Aplatir la sortie
        x = self.fc(x)  # Passer à la couche fully connected
        x = self.dropout(x)
        x = self.out(x) 

        return x
