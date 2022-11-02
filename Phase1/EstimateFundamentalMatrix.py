# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EstimateFundamentalMatrix.py

import numpy as np


def get_fundamental_mat(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    x_mat = np.zeros((len(points1), 9))
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        x1, y1 = p1
        x2, y2 = p2
        # x_mat[i, :] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1.0]
        # x_mat[i, :] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1.0]
        u = x1
        v = y1
        up = x2
        vp = y2
        # x_mat[i, :] = [u*up, u*vp, u, v*up, v*vp, v, up, vp, 1.0]
        # x_mat[i, :] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

        # Constructed with 8 points: [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
        # x_mat[i, :] = [x1 * x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        x_mat[i, :] = [x2 * x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # U, D, VT = np.linalg.svd(x_mat.T @ x_mat)
    # x = VT[-1]
    # F_hat = x.reshape((3, 3)).T
    # # F = F_hat
    # FU, FD, FVT = np.linalg.svd(F_hat)
    # FD[2] = 0
    # FD = np.diag(FD)
    # F = FU @ (FD @ FVT)
    # F /= F[2, 2]

    # Find the linear least square solution with SVD
    U, D, V = np.linalg.svd(x_mat)
    F_hat = V[-1].reshape(3, 3)

    # Constrain the F matrix by making it rank 2 by zeroing the last singular value
    U, D, V = np.linalg.svd(F_hat)
    D[-1] = 0
    F = np.dot(U, np.dot(np.diag(D), V))
    F /= F[2, 2]

    # import cv2
    # F2, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT, None, None, None)

    return F
