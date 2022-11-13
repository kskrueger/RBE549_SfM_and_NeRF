# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# BundleAdjustment.py

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import scipy.optimize

def BundleAdjustment(V, X, P, x, n_cams):
    def loss(params):
        X = params.reshape((-1,4))
        # X = params
        final_err = 0
        for idx in n_cams:
            err = (
                V[idx,:] * (np.square((X @ P[idx][0,:]) / (X @ P[idx][2,:]) - x[idx][:,0])) +
                V[idx,:] * (np.square((X @ P[idx][1,:]) / (X @ P[idx][2,:]) - x[idx][:,1]))
                ).sum()
            final_err += err
        return final_err
        # return (err1 + err2).sum()

    def loss(params):
        error = 0
        # n_images = len(image_paths)
        error += loss(params)
        print(error)
        return error

    cams_pose = [] # R and T of the 6 cameras
    params = np.concatenate([cams_pose.reshape(-1), X.reshape(-1)],axis=0)
    optimized = scipy.optimize.least_squares(loss, params, x_scale='jac', ftol=1e-4, method='trf',)
    cam_pose_optimized = optimized.x[: n_cams*7].reshape((n_cams, 3, 3)) #7 for x,y,z,quaternion
    X_optimized = optimized.x[n_cams*7, :].reshape(X.shape)

    print(X_optimized, cam_pose_optimized)
    return X_optimized, cam_pose_optimized