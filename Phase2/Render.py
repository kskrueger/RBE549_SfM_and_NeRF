# RBE549: Building Built in Minutes using NeRF
# Karter Krueger and Tript Sharma
# Render.py

import torch.nn as nn
import torch
import numpy as np

def get_rays(img_size, K, cam_world_T):
    h, w = img_size
    # Get u and v index lists the shape of the image (H x W)
    u, v = torch.meshgrid(torch.linspace(0, w-1, w),
                          torch.linspace(0, h-1, h))

    # Find ray directions using camera intrinsics K
    cx = K[0][2]
    fx = K[0][0]
    cy = K[1][2]
    fy = K[1][1]
    directions = torch.stack([(u - cx) / fx,
                            -(v - cy) / fy,
                            -torch.ones_like(u)], -1)

    # Rotate rays to world frame
    ray_dirs = torch.sum(directions[..., None, :] * cam_world_T[:3, :3], -1)

    # The camera origin applies to all rays using pin-hole model
    ray_origins = cam_world_T[:3, -1].expand(ray_dirs.shape)
    return torch.stack((ray_origins, ray_dirs))

# From paper: Once we shift to near plane, simple sample linearly from 0 to 1 for disparity n to inf original space
def shift_rays_ndc(img_size, K, near, ray_origins, ray_dirs):
    h, w = img_size
    fx = K[0, 0]
    fy = K[1, 1]

    # Shift ray origins to be on the "near" plane
    d = -(near + ray_origins[..., 2]) / ray_dirs[..., 2]
    ray_origins = ray_origins + d[..., None] * ray_dirs

    ox = ray_origins[..., 0]
    oy = ray_origins[..., 1]
    oz = ray_origins[..., 2]
    dx = ray_dirs[..., 0]
    dy = ray_dirs[..., 1]
    dz = ray_dirs[..., 2]

    # Project using equations (25) and (26)
    ox2 = -1. / (w / (2. * fx)) * ox / oz
    oy2 = -1. / (h / (2. * fy)) * oy / oz
    oz2 = 1. + 2. * near / oz

    dx2 = -1. / (w / (2. * fx)) * (dx / dz - ox / oz)
    dy2 = -1. / (h / (2. * fy)) * (dy / dz - oy / oz)
    dz2 = -2. * near / oz

    ray_origins = torch.stack([ox2, oy2, oz2], -1)
    ray_dirs = torch.stack([dx2, dy2, dz2], -1)

    return ray_origins, ray_dirs

# TODO: redo this!
# Volume sampling from paper (5.2)
# The math behind PDF and CDF is very similar to the github repo here: (https://github.com/yenchenlin/nerf-pytorch/)
# Professor Sanket gave permission to use this math after we explained the logic behind it to him during office hours
def volume_sampling(z_vals, weights, samples_per_ray):
    # From paper, EQ (5): Normalize weights to produce piecewise-constant PDF along the ray
    weights_norm = (weights + 1e-4) / torch.sum(weights, -1, keepdim=True)  # normalize weights (with small noise)
    # Cumulative sum along the weights ( w_i_hat = w_i / sum(w_j for j=1 to N_c))
    cumulative_sum = torch.cumsum(weights_norm, -1)
    # We now have a distribution where there are higher cdf weights *after* the first peak (which is why we will invert)
    cdf = torch.cat([torch.zeros_like(cumulative_sum[..., :1]), cumulative_sum], -1)  # (len(rays), len(samples))
    # Generate new steps for the fine network
    fine_steps = torch.linspace(0., 1., steps=samples_per_ray)
    # reshape/expand into size (num_rays, num_samples)
    fine_steps = fine_steps.expand([*cdf.shape[:-1], samples_per_ray])

    # Now invert the CDF to have the focus towards the dense areas rather than sparse and blocked areas
    # find the index values for the closest cdf values that correspond to fine steps
    nearest_idxs = torch.searchsorted(cdf, fine_steps.contiguous(), right=True)
    below_idx = torch.max(torch.zeros_like(nearest_idxs - 1), nearest_idxs - 1)
    above_idx = torch.min((cdf.shape[-1] - 1) * torch.ones_like(nearest_idxs), nearest_idxs)
    idx_bounds = torch.stack([below_idx, above_idx], -1)  # Size is now (num_rays, num_samples, 2)

    bound_shape = [idx_bounds.shape[0], idx_bounds.shape[1], cdf.shape[-1]]
    cdf_mm = torch.gather(cdf[:, None, ...].expand(bound_shape), 2, idx_bounds)  # collect min and max from each idx
    z_vals_mm = torch.gather(z_vals[:, None, ...].expand(bound_shape), 2, idx_bounds)

    cdf_mm_diffs = (cdf_mm[..., 1] - cdf_mm[..., 0])
    cdf_mm_diffs = torch.where(cdf_mm_diffs < 1e-5, torch.ones_like(cdf_mm_diffs), cdf_mm_diffs)
    t = (fine_steps - cdf_mm[..., 0]) / cdf_mm_diffs
    fine_samples = z_vals_mm[..., 0] + t * (z_vals_mm[..., 1] - z_vals_mm[..., 0])

    return fine_samples


def render(rays_batch, coarse_net, fine_net, xyz_embed, dir_embed, K, img_size, coarse_steps=64, near_dist=0., far_dist=1.):
    # convert rays to NDC space
    rays_orig, rays_dir = shift_rays_ndc(img_size, K, 1., rays_batch[:, 0], rays_batch[:, 1])
    rays_batch = torch.stack((rays_orig, rays_dir), -2)

    t_samples = torch.linspace(0, 1, coarse_steps)
    z_dists = (t_samples * far_dist)[..., None].expand((len(t_samples), 3))

    # convert to points using origin + vectors * dist
    pts = rays_batch[:, 0, None].expand((-1, coarse_steps, 3)) + rays_batch[:, 1, None].expand((-1, coarse_steps, 3)) * z_dists
    pts_flat = pts.reshape((-1, 3))

    view_dirs_flat = rays_batch[:, 1, None].expand((-1, coarse_steps, 3)).reshape(-1, 3)
    view_dirs_flat = view_dirs_flat / torch.norm(view_dirs_flat, dim=-1)[..., None].expand((-1, 3))

    pts_embedded = xyz_embed.embed(pts_flat).float()
    dir_embedded = dir_embed.embed(view_dirs_flat).float()

    coarse_outputs = coarse_net(pts_embedded, dir_embedded)
    coarse_rgb = nn.Sigmoid()(coarse_outputs[:, :3])

    diffs = torch.cat((torch.diff(z_dists[:, 0]), torch.tensor([1e10])))
    dists = diffs[None, :].expand((rays_batch.shape[0], coarse_steps)) * torch.norm(rays_batch[:, 1, None], dim=-1).expand((-1, coarse_steps))
    dists_flat = dists.reshape((-1))

    raw_noise_std = 1e0
    noise = torch.randn(coarse_outputs[..., 3].shape) * raw_noise_std
    coarse_alpha = 1 - torch.exp(-nn.ReLU()(coarse_outputs[:, 3] + noise) * dists_flat)

    weights = coarse_alpha * torch.cumprod(torch.cat((torch.tensor([1]), 1 - coarse_alpha), -1), -1)[..., :-1]

    pixels = torch.sum(weights.reshape((-1, coarse_steps))[..., None] * coarse_rgb.reshape((-1, coarse_steps, 3)), -2)

    return pixels
