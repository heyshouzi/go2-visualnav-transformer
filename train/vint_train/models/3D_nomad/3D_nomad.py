import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional
from pointnet import PointNet  # 假设 PointNet 类已经定义并可以导入
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding


class LiDAREncoder(nn.Module):
    def __init__(self, input_channels: int = 3, encoding_size: int = 128):
        super(LiDAREncoder, self).__init__()
        # 假设 PointNet 网络已经定义好
        self.pointnet = PointNet(input_channels=input_channels, output_channels=encoding_size)
        self.encoding_size = encoding_size

    def forward(self, lidar_points: torch.Tensor) -> torch.Tensor:
        # lidar_points 的形状是 [batch_size, num_points, 3]，这里假设输入是 [batch_size, num_points, 3]
        return self.pointnet(lidar_points)


class ThreeDNomad(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        lidar_encoding_size: Optional[int] = 512,  # 新增 LiDAR 编码大小
        lidar_encoder: Optional[nn.Module] = None, 
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # 视觉信息编码器
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError

        # 目标信息编码器
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # LiDAR 编码器
        self.lidar_encoder = LiDAREncoder(input_channels=3, encoding_size=lidar_encoding_size)

        # 编码器压缩层
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 位置编码与自注意力层
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        obs_lidar: torch.Tensor,  
        goal_lidar: torch.Tensor,
        lidar_mask: torch.Tensor = None,  # mask to choose which obs_lidar and goal_lidar data to use 
        input_goal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        device = obs_img.device

        # 目标编码
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)

        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # 目标图像编码
        obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=1)
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)

        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding

        # 获取观察图像编码
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)

        # LiDAR编码
        if lidar_mask is not None and lidar_mask.item() == 1:
            obs_lidar = torch.zeros_like(obs_lidar)
            goal_lidar = torch.zeros_like(goal_lidar)
        obs_lidar_encoding = self.lidar_encoder(obs_lidar)  # 处理 LiDAR 数据
        goal_lidar_encoding = self.lidar_encoder(goal_lidar)  # 处理目标 LiDAR 数据
        # 合并视觉信息与 LiDAR 信息
        combined_encoding = torch.cat((obs_encoding, obs_lidar_encoding,goal_lidar_encoding.unsqueeze(1)), dim=1)
        # 位置编码
        if self.positional_encoding:
            combined_encoding = self.positional_encoding(combined_encoding)

        # 自注意力处理
        obs_encoding_tokens = self.sa_encoder(combined_encoding)

        
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)


        if input_goal_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, goal_mask.long()).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask

        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens




# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module