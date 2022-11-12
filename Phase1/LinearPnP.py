# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# LinearPnP.py

import numpy as np


def linear_pnp(x: np.ndarray, X: np.ndarray, K) -> tuple[np.ndarray, np.ndarray]:
    # Solve with SVD for Ax=0, with A being a 2n x 12 matrix for n points
    A = np.zeros((2*len(x), 12))
    for i, (xi, Xi) in enumerate(zip(x, X)):
        xn, yn = xi
        Xn, Yn, Zn = Xi
        A[i*2, :] = [Xn, Yn, Zn, 1, 0, 0, 0, 0, -xn*Xn, -xn*Yn, -xn*Zn, -xn]
        A[i*2+1, :] = [0, 0, 0, 0, Xn, Yn, Zn, 1, -yn*Xn, -yn*Yn, -yn*Zn, -yn]

    # Find the linear least square solution with SVD
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape(3, 4)

    # Solve for R and T using K and P
    K_inv = np.linalg.inv(K)
    U, D, VT = np.linalg.svd(K_inv @ P[:, :3])
    R = U @ VT
    T = K_inv @ P[:, 3:] / D[0]

    return T, R
