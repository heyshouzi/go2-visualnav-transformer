import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# https://arxiv.org/pdf/2403.03954
class dp3Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs):
        super().__init__()

        block_channel = [64, 128, 256]

        assert in_channels == 3, f"dp3Net only supports 3 channels, but got {in_channels}"

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(int(block_channel[-1]), int(out_channels)),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(int(block_channel[-1]), int(out_channels))
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            print("[dp3Net] not use projection")

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4:  # (batch_size, context_size, n, 3) or (batch_size, 1, n, 3)
            batch_size, context_size, n, in_channels = x.shape
            x = x.view(batch_size * context_size, n, in_channels)  # (batch_size * context_size, n, 3)
            x = x.view(batch_size * context_size * n, in_channels)  # (batch_size * context_size * n, 3)
            x = self.mlp(x)  # (batch_size * context_size * n, out_channels)
            x = x.view(batch_size * context_size, n, -1)  # (batch_size * context_size, n, out_channels)
            x = torch.max(x, 1)[0]  # (batch_size * context_size, out_channels)
            x = x.view(batch_size, context_size, -1)  # (batch_size, context_size, out_channels)
        elif len(x.shape) == 3:  # (batch_size, n, 3)
            batch_size, n, in_channels = x.shape
            x = x.view(batch_size * n, in_channels)  # (batch_size * n, 3)
            x = self.mlp(x)  # (batch_size * n, out_channels)
            x = x.view(batch_size, n, -1)  # (batch_size, n, out_channels)
            x = torch.max(x, 1)[0]  # (batch_size, out_channels)
        else:
            raise ValueError("Input shape must be (batch_size, context_size, n, 3) or (batch_size, n, 3)")

        x = self.final_projection(x)  # (batch_size, context_size, out_channels) or (batch_size, out_channels)
        return x



# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
# https://arxiv.org/pdf/1612.00593

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x:torch.Tensor):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.tensor([1,0,0,0,1,0,0,0,1], dtype=torch.float32).view(1, 9).repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(PointNet,self).__init__()
        self.stn = STN3d(in_channels=in_channels, out_channels=out_channels)
        self.out_channels = out_channels
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_channels, 1)  # output channels are specified as out_channels
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)
  
    def forward(self, x:torch.Tensor):
        if len(x.shape) == 4:  # (batch_size, context_size, n, 3) included context size = 1 
            batch_size, context_size, n, in_channels = x.shape
            x = x.view(batch_size * context_size, n, in_channels)  # (batch_size * context_size, n, 3)
            x = x.transpose(2, 1)  # (batch_size * context_size, 3, n)
            trans = self.stn(x)  # (batch_size * context_size, 3, 3)
            x = torch.bmm(x, trans)  # (batch_size * context_size, 3, n)
            x = x.transpose(2, 1)  # (batch_size * context_size, n, 3)
            x = F.relu(self.bn1(self.conv1(x.transpose(2, 1))))  # (batch_size * context_size, 64, n)
            x = F.relu(self.bn2(self.conv2(x)))  # (batch_size * context_size, 128, n)
            x = self.bn3(self.conv3(x))  # (batch_size * context_size, out_channels, n)
            x = torch.max(x, 2, keepdim=True)[0]  # (batch_size * context_size, out_channels, 1)
            x = x.view(batch_size * context_size, self.out_channels)  # (batch_size * context_size, out_channels)
            x = x.view(batch_size, context_size, -1)  # (batch_size, context_size, out_channels)
        elif len(x.shape) == 3:  # (batch_size, n, 3)
            batch_size, n, in_channels = x.shape
            x = x.transpose(2, 1)  # (batch_size, 3, n)
            trans = self.stn(x)  # (batch_size, 3, 3)
            x = torch.bmm(x, trans)  # (batch_size, 3, n)
            x = x.transpose(2, 1)  # (batch_size, n, 3)
            x = F.relu(self.bn1(self.conv1(x.transpose(2, 1))))  # (batch_size, 64, n)
            x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, n)
            x = self.bn3(self.conv3(x))  # (batch_size, out_channels, n)
            x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, out_channels, 1)
            x = x.view(batch_size, self.out_channels)  # (batch_size, out_channels)
        else:
            raise ValueError("Input shape must be (batch_size, context_size, n, 3) or (batch_size, n, 3)")
        return x  # Return only the feature vector


# PointNet++: Deep Hierarchical Feature Learning onPoint Sets in a Metric Space 
#  https://arxiv.org/pdf/1706.02413
class LocalFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(LocalFeatureExtractor, self).__init__()
        # The input is `in_channels` number of features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),  # The input has `in_channels` (3D coordinates + other features)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
    
    def forward(self, center, neighbors):
        # Concatenate the center point with its neighboring points
        features = torch.cat([center, neighbors], dim=-1)  # Combine the features
        return self.mlp(features)  # Return local features

class PointNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(PointNetPlusPlus, self).__init__()
        
        # Initialize the local feature extraction module, with flexible input dimensions
        self.local_feature_extractor = LocalFeatureExtractor(in_channels)
        
        # Fully connected layers to output the feature vector
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)  # Output the feature vector, with dimensionality defined by `out_channels`
        
    def forward(self, x):
        # Assume the input `x` has the shape (batch_size, N, in_channels), 
        # where N is the number of points, and `in_channels` is the feature dimension per point
        batch_size, num_points, _ = x.size()
        
        # Step 1: Farthest Point Sampling (FPS) to select representative points (this is a simplified assumption)
        centers = x[:, :1, :]  # Assume we choose the first point as the center
        
        # Step 2: Select neighboring points for each center point (simplified assumption)
        neighbors = x[:, 1:4, :]  # Assume we choose the next 3 points as neighbors
        
        # Step 3: Local feature extraction
        local_features = self.local_feature_extractor(centers, neighbors)
        
        # Step 4: Global feature aggregation using max pooling
        global_feature = torch.max(local_features, dim=1)[0]  # Max pooling across local features
        
        # Step 5: Fully connected layers to produce the feature vector
        x = F.relu(self.fc1(global_feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output the final feature vector
        return x  # Return the feature vector (shape: batch_size x out_channels)

#https://arxiv.org/pdf/2012.09164
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