# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EssentialMatrixFromFundamentalMatrix.py
import numpy as np


def get_E_from_F(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    E = K.T @ F @ K

    U, D, VT = np.linalg.svd(E)
    D[:] = [1, 1, 0]
    E = U @ (np.diag(D) @ VT)
    return E
