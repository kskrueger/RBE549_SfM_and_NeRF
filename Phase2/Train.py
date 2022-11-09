# RBE549: Building Built in Minutes using NeRF
# Karter Krueger and Tript Sharma
# Train.py

import torch
import torch.nn as nn
import numpy as np

from Loader import LoadData
from Network import NeRFNet, Encoder
from Render import get_rays, render

EPOCHS = 200000
NUM_RAYS = 10000
device = 'gpu' if torch.cuda.is_available() else 'cpu'

data_path = "/Users/kskrueger/Projects/RBE549_SfM_and_NeRF/Phase2/Data/lego/"
K, train_imgs, train_T, test_imgs, test_T, val_imgs, val_T = LoadData(data_path, device=device)
H, W = train_imgs[0].shape[:2]

coarse_net = NeRFNet()
coarse_net.to(device)

fine_net = NeRFNet()
fine_net.to(device)

opt = torch.optim.Adam(coarse_net.parameters(), lr=1e-4)

xyz_embed = Encoder(10)
dir_embed = Encoder(4)

train_rays = torch.stack([get_rays((H, W), K, cam_T) for cam_T in train_T])
train_rays_flat = train_rays.permute((0, 2, 3, 1, 4)).reshape((train_rays.shape[0]*H*W, 2, 3))
train_pixels_flat = train_imgs.reshape((train_imgs.shape[0]*H*W, 3))

for e in range(EPOCHS):
    idx = np.random.randint(0, train_rays_flat.shape[0], NUM_RAYS)
    train_rays_batch = train_rays_flat[idx]
    pixels_batch = train_pixels_flat[idx]

    ray_rgbs = render(train_rays_batch, coarse_net, fine_net, xyz_embed, dir_embed)

    loss = torch.mean(torch.square(ray_rgbs - pixels_batch))

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Epoch", e, "Loss: ", loss.detach().cpu().numpy())

    # Save checkpoint every some SaveCheckPoint's iterations
    if e % 5000 == 0:
        # Save the Model learnt in this epoch

        torch.save(
            {
                "epoch": e,
                "model_state_dict": coarse_net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss.detach().cpu().numpy(),
            },
            "latest.ckpt",
        )
        print("\n" + " Model Saved...")


