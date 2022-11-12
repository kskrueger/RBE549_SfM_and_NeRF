# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# NonlinearTriangulation.py

import numpy as np
import scipy.optimize


def nonlinear_triangulate(X_pts, P1, P2, pts1, pts2, K):
    # P1 = np.hstack((P1[1], P1[1] @ -P1[0]))
    # P2 = np.hstack((P2[1], P2[1] @ -P2[0]))
    P1 = np.dot(K, np.dot(P1[1], np.hstack((np.eye(3), -P1[0]))))
    P2 = np.dot(K, np.dot(P2[1], np.hstack((np.eye(3), -P2[0]))))

    def error_fn(params):
        X = params.reshape((len(X_pts), 3))
        Xh = np.hstack([X, np.ones((len(X), 1))])

        error = 0
        for P, pts in zip([P1, P2], [pts1, pts2]):
            error = np.abs(pts[:, 0] - ((P[0].T @ Xh.T) / (P[2].T @ Xh.T))) \
                    + np.abs(pts[:, 1] - ((P[1].T @ Xh.T) / (P[2].T @ Xh.T)))

        total_error = np.sum(error)
        # print("Error: ", total_error)
        return total_error

    x0 = X_pts.flatten()

    # x = arg min(sum(func(y)**2,axis=0))
    #          y
    print("Running NonLinearTriangulation Optimization... This may take a bit")
    results = scipy.optimize.least_squares(error_fn, x0, method='dogbox')  # , options={'maxiter': 1000})
    print("NonLinearTriangulation Optimization: Done! (FinalCost: {})".format(results.fun))
    return results.x.reshape(X_pts.shape)
