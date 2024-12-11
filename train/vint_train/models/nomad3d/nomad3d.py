import torch.nn as nn

class NoMaD3D(nn.Module):

    def __init__(self, vision_lidar_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(NoMaD3D, self).__init__()


        self.vision_lidar_encoder = vision_lidar_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_lidar_encoder" :
            output = self.vision_lidar_encoder(kwargs["obs_img"], 
                                               kwargs["goal_img"], 
                                               kwargs["obs_lidar"],
                                               kwargs["goal_lidar"],
                                               lidar_mask=kwargs["lidar_mask"],
                                               input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], 
                                         timestep=kwargs["timestep"],
                                        global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork_3d(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork_3d, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output



