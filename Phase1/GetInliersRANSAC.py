# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# GetInliersRANSAC.py
import cv2

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

        # out = cv2.drawMatches(imgs[0],
        #                       [cv2.KeyPoint(int(pt1[0]), int(pt1[1]), 1) for pt1 in pts1],
        #                       imgs[1],
        #                       [cv2.KeyPoint(int(pt2[0]), int(pt2[1]), 1) for pt2 in pts2],
        #                       [cv2.DMatch(i, i, 1) for i in np.arange(len(pts1))], None)
        # print(pts1 - pts2)
        # cv2.imshow("Out", out)
        # cv2.waitKey(5)

        F = get_fundamental_mat(pts1, pts2)
        # Fcv, inliers_cv = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT, inlier_thresh, 1, max_iters)
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

        # print("F", F)
        # print("avg_error", np.mean(np.abs(inlier_errors)))
        # print("num_inliers", num_inliers)
        # print()

    final_F = get_fundamental_mat(points1[best_inliers_idx[:, 0]][:, 0:2], points2[best_inliers_idx[:, 0]][:, 0:2])
    return final_F, best_inliers_idx
