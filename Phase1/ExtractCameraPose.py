# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# ExtractCameraPose.py

import numpy as np

def get_camera_poses(E):
    U, D, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1, R1 = U[:, 2:], (U @ W @ VT)
    C2, R2 = -U[:, 2:], (U @ W @ VT)
    C3, R3 = U[:, 2:], (U @ W.T @ VT)
    C4, R4 = -U[:, 2:], (U @ W.T @ VT)
    poses = [[C1, R1], [C2, R2], [C3, R3], [C4, R4]]
    poses_out = []
    for (c, r) in poses:
        if np.linalg.det(r) < 0:
            c = -c
            r = -r
        poses_out.append((c, r))

    # Pose: P = K @ R @ np.hstack([np.eye(3), -C])

    return poses_out
