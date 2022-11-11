# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# GetInliersRANSAC.py

from EstimateFundamentalMatrix import get_fundamental_mat
import numpy as np


def ransac_F(points1: np.ndarray, points2: np.ndarray, inlier_thresh: float, max_iters: int, imgs=None) -> tuple[np.ndarray, np.ndarray]:
    max_inliers = 0
    best_inliers_idx = np.zeros((len(points1)))
    points1 = np.hstack((points1, np.ones((len(points1), 1))))
    points2 = np.hstack((points2, np.ones((len(points2), 1))))
    for i in range(max_iters):
        rand_pt_idx = np.random.randint(0, len(points1), (8))
        pts1 = points1[rand_pt_idx, 0:2]
        pts2 = points2[rand_pt_idx, 0:2]

        F = get_fundamental_mat(pts1, pts2)
        inlier_errors = np.zeros((len(points1), 1))
        for j in range(len(points1)):
            inlier_errors[j] = points2[j].T @ (F @ points1[j])

        inliers_idx = np.abs(inlier_errors) < inlier_thresh
        num_inliers = np.sum(inliers_idx)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_idx = inliers_idx
            best_errors = inlier_errors
            best_F = F

    print("RansacF max_inliers pre:", max_inliers)
    print("RansacF pre F:", best_F)

    final_F = get_fundamental_mat(points1[best_inliers_idx[:, 0]][:, 0:2], points2[best_inliers_idx[:, 0]][:, 0:2])
    final_inlier_errors = np.zeros((len(points1), 1))
    for j in range(len(points1)):
        final_inlier_errors[j] = points2[j].T @ (final_F @ points1[j])
    final_inliers_idx = np.abs(final_inlier_errors) < inlier_thresh
    final_num_inliers = np.sum(final_inliers_idx)
    print("RansacF max_inliers post:", final_num_inliers)
    print("RansacF post F:", final_F)

    return final_F, final_inliers_idx
