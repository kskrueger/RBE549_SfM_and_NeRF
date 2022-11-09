# RBE549: Building Built in Minutes using NeRF
# Karter Krueger and Tript Sharma
# Network.py

import torch.nn as nn
import torch


# NeRF frequency positional encodings generator
class Encoder:
    def __init__(self, L_encodings):
        self.L_encodings = L_encodings

        embeddings = []
        dims = 3
        # embeddings.append(lambda x: x)
        periodic_functions = [torch.sin, torch.cos]

        # Sample using Log
        frequency_bands = 2.0 ** torch.arange(0, self.L_encodings)  # From 0 to L-1
        for freq in frequency_bands:
            for gamma_fn in periodic_functions:
                embeddings.append(lambda x: gamma_fn(x * freq * torch.pi))  # TODO: add pi here?
        out_size = (len(embeddings)) * dims

        self.embeddings = embeddings
        self.out_size = out_size

    def embed(self, inputs):
        return torch.cat([func(inputs) for func in self.embeddings], -1)


class NeRFNet(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """

        super().__init__()

        network_width = 256
        xyz_features_size = 3 * 20
        dir_features_size = 2 * 12

        self.fc1_xyz = nn.Linear(xyz_features_size, network_width)  # input xyz layer
        self.fc2 = nn.Linear(network_width, network_width)  # layer 2
        self.fc3 = nn.Linear(network_width, network_width)  # layer 3
        self.fc4 = nn.Linear(network_width, network_width)  # layer 4
        self.fc5_xyz = nn.Linear(network_width + xyz_features_size, network_width)  # layer 5, concatenate xyz again
        self.fc6 = nn.Linear(network_width, network_width)  # layer 6
        self.fc7 = nn.Linear(network_width, network_width)  # layer 7
        self.fc8 = nn.Linear(network_width, network_width)  # last layer 8
        self.fc_alpha = nn.Linear(network_width, 1)
        self.alpha_relu = nn.ReLU()  # apply ReLU to fc_alpha to get volume density
        self.fc_rgb1 = nn.Linear(network_width, network_width)  # concatenate direction before rgb
        self.fc_rgb2 = nn.Linear(network_width + dir_features_size, network_width//2)
        self.fc_rgb3 = nn.Linear(network_width//2, 3)
        self.rgb_sigmoid = nn.Sigmoid()  # apply Sigmoid to fc_rgb2 to get rgb values

    def forward(self, input_xyz, input_dir):
        """
        Input:
        x_input is a MiniBatch of rays
        Outputs:
        out - *UNACTIVATED* output of the network
        """
        x = self.fc1_xyz(input_xyz)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x4 = self.fc4(x)
        x4 = nn.ReLU()(x4)
        x_xyz = torch.concat([x4, input_xyz], -1)
        x = self.fc5_xyz(x_xyz)
        x = nn.ReLU()(x)
        x = self.fc6(x)
        x = nn.ReLU()(x)
        x = self.fc7(x)
        x = nn.ReLU()(x)
        x8 = self.fc8(x)
        x8 = nn.ReLU()(x8)

        alpha = self.fc_alpha(x8)
        # alpha = self.alpha_relu(alpha)

        rgb_fc = self.fc_rgb1(x8)  # 256 x 256
        rgb_fc = nn.ReLU()(rgb_fc)
        x_dir = torch.cat([rgb_fc, input_dir], -1)
        rgb_fc = self.fc_rgb2(x_dir)  # 280 x 128
        rgb_fc = nn.ReLU()(rgb_fc)
        rgb = self.fc_rgb3(rgb_fc)  # 128 x 3
        # rgb = self.rgb_sigmoid(rgb_fc)

        output = torch.cat([rgb, alpha], -1)
        return output
