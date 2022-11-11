# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# PnPRANSAC.py

import numpy as np
from LinearPnP import linear_pnp


def ransac_pnp(x: np.ndarray, X: np.ndarray, K, inlier_thresh: float, max_iters: int):
    max_inliers = 0
    best_inliers_idx = np.zeros((len(x)))
    best_P = None
    # points1 = np.hstack((, np.ones((len(points1), 1))))
    # points2 = np.hstack((points2, np.ones((len(points2), 1))))
    assert len(x) == len(X), "X must be same size as x"

    Xh = np.hstack((X, np.ones((len(X), 1))))
    u, v = x.T

    for i in range(max_iters):
        rand_pt_idx = np.random.randint(0, len(x), (6))
        xr = x[rand_pt_idx]
        Xr = X[rand_pt_idx]

        C, R = linear_pnp(xr, Xr, K)

        P = np.dot(K, np.dot(R, np.hstack((np.eye(3), -C))))
        error = np.square(u - ((P[0].T @ Xh.T) / (P[2].T @ Xh.T))) \
                + np.square(v - ((P[1].T @ Xh.T) / (P[2].T @ Xh.T)))

        inliers_idx = np.abs(error) < inlier_thresh
        num_inliers = np.sum(inliers_idx)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_idx = inliers_idx
            best_errors = error
            best_P = P

        # print("F", F)
        # print("avg_error", np.mean(np.abs(inlier_errors)))
        # print("num_inliers", num_inliers)
        # print()

    print("RansacPnp max_inliers Pre:", max_inliers)
    print("Pre P:", best_P)
    final_C, final_R = linear_pnp(x[best_inliers_idx], X[best_inliers_idx], K)
    u, v = x.T
    final_P = np.dot(K, np.dot(final_R, np.hstack((np.eye(3), -final_C))))
    error = np.square(u - ((final_P[0].T @ Xh.T) / (final_P[2].T @ Xh.T))) \
            + np.square(v - ((final_P[1].T @ Xh.T) / (final_P[2].T @ Xh.T)))
    inliers_idx = np.abs(error) < inlier_thresh
    num_inliers = np.sum(inliers_idx)
    print("RansacPnp max_inliers post:", num_inliers)
    print("RansacPnp Post P:", final_P)

    return final_P, final_C, final_R, best_inliers_idx

