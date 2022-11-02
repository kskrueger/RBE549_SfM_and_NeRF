# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# LinearTriangulation.py

import numpy as np


def skew_sym(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def linear_triangulate(pose1: tuple, pose2: tuple, points1: np.ndarray, points2: np.ndarray, K: np.ndarray):
    # P1 = K @ np.hstack([pose1[1], pose1[0]])
    # P2 = K @ np.hstack([pose2[1], pose2[0]])

    R1, C1 = pose1[1], pose1[0]
    R2, C2 = pose2[1], pose2[0]
    I = np.eye(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))
    # P1 = K @ np.hstack([pose1[1], pose1[1] @ -pose1[0]])
    # P2 = K @ np.hstack([pose2[1], pose2[1] @ -pose2[0]])

    Xs = np.zeros((len(points1), 3))

    for i, (p1, p2) in enumerate(zip(points1, points2)):
        # X1 = np.hstack([p1, 1]).T
        # X2 = np.hstack([p2, 1]).T
        X1 = skew_sym(p1)
        X2 = skew_sym(p2)
        pre_x = np.vstack([X1 @ P1,
                           X2 @ P2])
        U, D, VT = np.linalg.svd(pre_x)
        X = VT[-1]
        X /= X[-1]
        Xs[i, :] = X[:3]

    return Xs
