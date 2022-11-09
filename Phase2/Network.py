import torch
import torch.nn as nn
import numpy as np

class NeRF(nn.Module):
    '''
    class containing abstract architecture of NeRF for coarse and fine network training
    '''
    def __init__(self, position_encoder_size, direction_encoder_size):
        self.first_half_network = nn.Sequential(
            nn.Linear(in_features=position_encoder_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
        )
        # add skip connection to the activation of fifth layer
        self.sixth_layer_network = nn.Linear(in_features=position_encoder_size+256,out_features=256)
        self.second_half_network = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            # nn.ReLU(),
        )
        self.volume_density_layer = nn.Sequential(
            nn.Linear(in_features=256,out_features=1),
            nn.ReLU()
        )
        self.rgb_network = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
        )
        self.rgb_network_final_layer = nn.Sequential(
            nn.Linear(in_features=256+direction_encoder_size, out_features=128),
            nn.Sigmoid()
        )

    def forward(self, gammaX, gammaD):
        _result = self.first_half_network(gammaX)
        residual = torch.cat([gammaX, _result], -1)
        _result = self.sixth_layer_network(residual)
        _result = self.second_half_network(_result)
        
        volume_density = self.volume_density_layer(_result)

        _rgb_network = self.rgb_network(_result)
        rgb_final_layer_input = torch.cat([gammaD, _rgb_network], -1)
        rgb = self.rgb_network_final_layer(rgb_final_layer_input)

        return volume_density, rgb