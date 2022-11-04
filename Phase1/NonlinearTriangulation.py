# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# NonlinearTriangulation.py
import scipy as sc
import numpy as np

def nonLinearTriangulation(X_init, P1, P2, u1, u2):
    '''
    Reduce geometric error of projected points from Linear Triangulation using Least squares optimization
    Inputs:
        X  = Homogenous world coordinates' initial guess (Linear triangulation output)
        P1 = Projection matrix for img1
        P2 = Projection matrix for img2
        u1 = image1 coordinates
        u2 = image2 coordinates
        K  = camera intrisics
    Outputs:
        optimized X
    '''
    def geometric_loss(params):
        X = params.reshape((-1,4))
        return (
            (np.square((X @ P1[0,:]) / (X @ P1[2,:]) - u1[:,0]) + 
            np.square((X @ P1[0,:]) / (X @ P1[2,:]) - u1[:,1]))
            +
            (np.square((X @ P2[0,:]) / (X @ P2[2,:]) - u2[:,0]) + 
            np.square((X @ P2[1,:]) / (X @ P2[2,:]) - u2[:,1]))
            ).sum()
        # return (err1 + err2).sum()

    def minimization_fn(params):
        error = 0
        # n_images = len(image_paths)
        error += geometric_loss(params)
        print(error)
        return error

    params = X_init.reshape(-1)
    optimized = sc.optimize.least_squares(minimization_fn, params, method='dogbox', ftol=100000000, gtol=100, max_nfev=1000)#, options={'maxiter':1000})
    # params = [X_init]
    # optimized = sc.optimize.minimize(minimization_fn, params, options={'maxiter':1000})
    print(optimized)

    X_optimized = optimized.x
    X_optimized = X_optimized.reshape((-1,4))
    # X_optimized /= X_optimized[:,-1]

    print(X_optimized)
    return X_optimized