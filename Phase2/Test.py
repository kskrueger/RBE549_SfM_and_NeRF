# RBE549: Building Built in Minutes using NeRF
# Karter Krueger and Tript Sharma
# Train.py

import torch
import torch.nn as nn
import numpy as np

from Loader import LoadData
from Network import NeRFNet, Encoder
from Render import get_rays, render

from torch.utils.tensorboard import SummaryWriter

NUM_RAYS = 6000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "/media/karterk/GeneralSSD/Classes/RBE549_CV/NeRF_SfM/Phase2/Data/lego"
K, train_imgs, train_T, test_imgs, test_T, val_imgs, val_T = LoadData(data_path, device='cpu')
H, W = train_imgs[0].shape[:2]

coarse_net = NeRFNet()
coarse_net.to(device)

ModelPath = "/media/karterk/GeneralSSD/Classes/RBE549_CV/NeRF_SfM/Phase2/latest_new1.ckpt"
ModelPath = "/media/karterk/GeneralSSD/Classes/RBE549_CV/NeRF_SfM/Phase2/latest.ckpt"
CheckPoint = torch.load(ModelPath)
coarse_net.load_state_dict(CheckPoint["model_state_dict"])

# opt = torch.optim.Adam(coarse_net.parameters(), lr=5e-4)

Writer = SummaryWriter("./Logs/")

xyz_embed = Encoder(10)
dir_embed = Encoder(4)

K = torch.from_numpy(K).cpu()

train_rays = torch.stack([get_rays((H, W), K, cam_T) for cam_T in train_T])
# if RAND_BATCHING:
#     train_rays_flat = train_rays.reshape((train_rays.shape[0]*H*W, 2, 3))
#     train_pixels_flat = train_imgs.reshape((train_imgs.shape[0]*H*W, 3))
#
#     rand_perm_idx = torch.randperm(train_rays_flat.shape[0])
#     train_rays_flat = train_rays_flat[rand_perm_idx]
#     train_pixels_flat = train_pixels_flat[rand_perm_idx]
#     idx = 0
# else:
train_rays_flat = train_rays.reshape((train_rays.shape[0], H*W, 2, 3))
train_pixels_flat = train_imgs.reshape((train_imgs.shape[0], H*W, 3))

hs = 0#int(H*.25)
he = H#int(H*.75)
ws = 0#int(W*.25)
we = W#int(H*.75)
H = he - hs
W = we - ws
train_rays_flat = train_rays[:, hs:he, ws:we].reshape((train_rays[:].shape[0], H*W, 2, 3))
train_pixels_flat = train_imgs[:, hs:he, ws:we].reshape((train_imgs[:].shape[0], H*W, 3))

output_img = torch.zeros((H, W, 3), dtype=torch.float).reshape((H*W, 3))
for e in range(1000):
    print("e", e)
    idx = np.random.choice(train_rays_flat.shape[0])
    rand_idx = np.random.choice(train_rays_flat.shape[1], size=(NUM_RAYS))
    train_rays_batch = train_rays_flat[idx, rand_idx].cuda()
    pixels_batch = train_pixels_flat[idx, rand_idx].cuda()

    with torch.no_grad():
        ray_rgbs = render(train_rays_batch, coarse_net, None, xyz_embed, dir_embed, K, (H, W))
        output_img[rand_idx] = ray_rgbs.float().cpu()

    # loss = nn.MSELoss()(ray_rgbs, pixels_batch)

    # loss_np = loss.detach().cpu().numpy()

import cv2
cv2.imwrite("out.png", (output_img.reshape(H, W, 3).detach().cpu().numpy() * 255).astype(np.uint8))
