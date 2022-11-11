import torch
import numpy as np
## function in torch
def generate_rays(img_height, img_width, K, c2w):
    '''
    get all the coordinates in the image. 
    generate directions from camera center to each point
        (u - cx)/fx, (v - cy)/fy
    rotate ray directions from cam to world frame (multiply with cam 2 world transformation)
        c2w.dot(ray_dir)
    translate cam frame's ray origin to world frame (origin of all rays)
    i.e. set origin to the translation vec in c2w
    '''
    u,v = torch.meshgrid(torch.arange(0,img_width),torch.arange(0,img_height))
    print(u,v)
    directions = torch.stack([(u-K[0][2])/K[0,0],-(v-K[1][2]/K[1,1]),-torch.ones_like(u)], 1)

    rays_dir = torch.sum(directions[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_org = c2w[:3,-1].expand(rays_dir.shape)

    return rays_org, rays_dir

def generate_ndc_rays(img_height, img_width, F, near, rays_org, rays_dir):
    '''
    NDC is the normalized device cooridinate. Projected points will be in normalized device coordinate (NDC) space, where
    the original viewing frustum mapped to the cube [−1, 1].
    implementing eq 25,26 from NeRF paper

    shift each ray's origin to near plane using 3D perspective projection matrix
    get the origin projection in the shifted image plane
    get the direction projetion in the shifted image plane
    '''
    translation = -(near + rays_org[:,2])/rays_dir[:,2]
    
    #origin projection in NDC space
    o_x = -F/(img_width/2) * (rays_org[:,0]/rays_org[:,2])
    o_y = -F/(img_height/2) * (rays_org[:,1]/rays_org[:,2])
    o_z = 1 + (2*near)/rays_org[:,2]

    #direction projection in NDC space
    d_x = -F/(img_width/2) * (rays_dir[:,0]/rays_dir[:,2] - rays_org[:,0]/rays_org[:,2])
    d_y = -F/(img_height/2) * (rays_dir[:,1]/rays_dir[:,2] - rays_org[:,1]/rays_org[:,2])
    d_z = -1 * (2*near)/rays_org[:,2]

    ndc_org = torch.stack([o_x, o_y, o_z], -1)
    ndc_dir = torch.stack([d_x, d_y, d_z], -1)

    return ndc_org, ndc_dir