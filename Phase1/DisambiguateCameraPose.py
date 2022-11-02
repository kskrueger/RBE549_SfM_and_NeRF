# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# DisambiguateCameraPose.py

import numpy as np
from LinearTriangulation import linear_triangulate


def disambiguate_pose(P0, poses_list, points1, points2, K):
    # TODO: delete
    from delete_helper import linear_triangulation_hartley


    if P0 is None:
        C0, R0 = np.zeros((3, 1)), np.eye(3)
        P0 = (C0, R0)

    max_valid = 0
    best_P = None
    best_X = None
    all_X = []
    for C, R in poses_list:
        r3 = R[-1]
        X = linear_triangulate(P0, (C, R), points1, points2, K)
        # TODO: delete
        # m1 = np.hstack([P0[1], P0[0]])
        # m2 = np.hstack([R, C])
        # X = linear_triangulation_hartley(points1.T, points2.T, m1, m2).T[:, :3]

        valid_idx = np.where(r3 @ (X - C.T).T > 0)
        all_X.append(X)
        if len(valid_idx) > max_valid:
            max_valid = len(valid_idx)
            best_P = (C, R)
            best_X = X

    return best_P, best_X, all_X
