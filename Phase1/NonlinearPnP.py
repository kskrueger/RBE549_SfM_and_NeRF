# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# NonlinearPnP.py

import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation


def nonlinear_pnp(C0: np.ndarray, R0: np.ndarray, x: np.ndarray, X: np.ndarray, K):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    # TODO: consider minimizing [quaternion, C] instead of [R, C]

    def error_fn(params):
        C = params[:3].reshape((3, 1))
        R = Rotation.from_quat(np.array(params[3:]) / 1).as_matrix()
        P = np.dot(K, np.dot(R, np.hstack((np.eye(3), -C))))

        error = np.square(x[:, 0] - ((P[0].T @ Xh.T) / (P[2].T @ Xh.T))) \
                + np.square(x[:, 1] - ((P[1].T @ Xh.T) / (P[2].T @ Xh.T)))

        total_error = np.sum(error)
        # print("TotalError", total_error)
        return total_error

    quat = Rotation.from_matrix(R0).as_quat() * 1
    params = [*C0.flatten(), *quat]

    # x = arg min(sum(func(y)**2,axis=0))
    #          y
    print("Running NonLinearPnP Optimization... This may take a bit")
    results = scipy.optimize.least_squares(error_fn, params, method='dogbox')  # , options={'maxiter': 1000})
    print("NonLinearPnP Optimization: Done! (FinalCost: {})".format(results.fun))
    params_out = results.x
    C = params_out[:3].reshape((3, 1))
    R = Rotation.from_quat(np.array(params_out[3:]) / 1).as_matrix()
    P = np.dot(K, np.dot(R, np.hstack((np.eye(3), -C))))

    return C, R, P
