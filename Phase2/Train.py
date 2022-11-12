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
NUM_RAYS = 100
RAND_BATCHING = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "/Users/kskrueger/Projects/RBE549_SfM_and_NeRF/Phase2/Data/lego"
K, train_imgs, train_T, test_imgs, test_T, val_imgs, val_T = LoadData(data_path, device='cpu')
H, W = train_imgs[0].shape[:2]

coarse_net = NeRFNet()
coarse_net.to(device)

fine_net = NeRFNet()
fine_net.to(device)

opt = torch.optim.Adam(coarse_net.parameters(), lr=.0001)

xyz_embed = Encoder(10)
dir_embed = Encoder(4)

K = torch.from_numpy(K).cpu()

train_rays = torch.stack([get_rays((H, W), K, cam_T) for cam_T in train_T])
if RAND_BATCHING:
    train_rays_flat = train_rays.reshape((train_rays.shape[0]*H*W, 2, 3))
    train_pixels_flat = train_imgs.reshape((train_imgs.shape[0]*H*W, 3))

    rand_perm_idx = torch.randperm(train_rays_flat.shape[0])
    train_rays_flat = train_rays_flat[rand_perm_idx]
    train_pixels_flat = train_pixels_flat[rand_perm_idx]
    idx = 0
else:
    train_rays_flat = train_rays.reshape((train_rays.shape[0], H*W, 2, 3))
    train_pixels_flat = train_imgs.reshape((train_imgs.shape[0], H*W, 3))

    hs = int(H*.25)
    he = int(H*.75)
    ws = int(W*.25)
    we = int(H*.75)
    H = he - hs
    W = we - ws
    train_rays_flat = train_rays[:1, hs:he, ws:we].reshape((train_rays[:1].shape[0], H*W, 2, 3))
    train_pixels_flat = train_imgs[:1, hs:he, ws:we].reshape((train_imgs[:1].shape[0], H*W, 3))

for e in range(EPOCHS):
    opt.zero_grad()
    # idx = np.random.randint(0, train_rays_flat.shape[0], NUM_RAYS)
    if RAND_BATCHING:
        train_rays_batch = train_rays_flat[idx:idx+NUM_RAYS]
        pixels_batch = train_pixels_flat[idx:idx+NUM_RAYS]
        idx += NUM_RAYS
        if (idx - NUM_RAYS) > len(train_pixels_flat) - 1:
            idx = 0
            rand_perm_idx = torch.randperm(train_rays_flat.shape[0])
            train_rays_flat = train_rays_flat[rand_perm_idx]
            train_pixels_flat = train_pixels_flat[rand_perm_idx]

    else:
        idx = np.random.choice(train_rays_flat.shape[0])
        rand_idx = np.random.choice(train_rays_flat.shape[1], size=(NUM_RAYS))
        train_rays_batch = train_rays_flat[idx, rand_idx]
        pixels_batch = train_pixels_flat[idx, rand_idx]

    ray_rgbs = render(train_rays_batch, coarse_net, fine_net, xyz_embed, dir_embed, K, (H, W))

    loss = nn.MSELoss()(ray_rgbs, pixels_batch)

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


