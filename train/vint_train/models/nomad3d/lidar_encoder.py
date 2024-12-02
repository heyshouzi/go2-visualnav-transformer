import torch
import torch.nn as nn
import torch.nn.functional as F

class dp3Net(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(dp3Net, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_channels, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class PointNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_channels, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x



class PointNetplusplus(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointNetplusplus, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

class PointTransformer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_channels, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (L, N, C) -> (N, C, L)
        x = self.fc(x)
        return x