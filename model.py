import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader




class CnnNet(nn.Module):
    def __init__(self, num_class=10):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1)
        )

        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(160, num_class)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        logit = self.activation(logit)

        return logit


class EmgDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]