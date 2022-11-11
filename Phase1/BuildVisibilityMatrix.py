# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# BuildVisibilityMatrix.py

import numpy as np
from scipy.sparse import lil_matrix


def build_visibility(visibility_matrix, ):
    # numCameras, numPoints, cameraIndices, pointIndices
    num_cameras = visibility_matrix.shape[0]
    num_features = visibility_matrix.shape[1]
    m = num_cameras * 2
    n = num_cameras * 7 + num_features * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(num_cameras)
    for s in range(7):
        A[2 * i, cameraIndices * 7 + s] = 1
        A[2 * i + 1, cameraIndices * 7 + s] = 1

    for s in range(3):
        A[2 * i, num_cameras * 7 + pointIndices * 3 + s] = 1
        A[2 * i + 1, num_cameras * 7 + pointIndices * 3 + s] = 1

    return A


