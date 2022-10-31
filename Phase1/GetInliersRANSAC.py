# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# GetInliersRANSAC.py
import numpy as np
from EstimateFundamentalMatrix import *
def getInliersRANSAC(pt_correspondences, eps, n_iterations):
    '''
    Inputs:
        set of all point correspondences between two images
        eps: threshold for inlier check
        n_iterations: max num of iterations
    Outputs:
        optimized F matrix

    for i in (1,n_iternatiion)
        get 8 random point pairs
        calculate F
        for j in (1, num_feature_pairs)
            if x2_j.T @ F @ x1_j < eps
                inlier_count++
                inliers.append()
        if(inlier_count>global_inlier_count)
            global_inlier_count = inlier_count
            global_inliers = inliers
        compute F again with global inliers
    return optimised F and indices of the best inlier correspondeces 
    '''
    ##
    global_inlier_count = 0
    global_inliers = []
    for i in range(n_iterations):
        #init local variables
        inlier_count = 0
        inliers = []
        #get 8 random point correspondences
        idx = np.random.randint(0,pt_correspondences.shape[0],8)
        eight_pt_correspondences = pt_correspondences[idx,:]
        #estimate the Fundamental Matrix
        F = estimateFundamentalMatrix(eight_pt_correspondences)
        for i, pt_pair in enumerate(pt_correspondences):
            x1,y1,x2,y2 = pt_pair[3:]
            X1 = np.array([x1,y1,1])
            X2 = np.array([x2,y2,1])
            
            val = X2.T @ F @ X1
            # print(val)
            if (abs(val) < eps):
                inlier_count+=1
                inliers.append(i)
        
        if(inlier_count>global_inlier_count):
            global_inlier_count=inlier_count
            global_inliers = inliers
    #optimize F with the best inlier set
    F_optimized = estimateFundamentalMatrix(pt_correspondences[global_inliers,:])
    # print(F_optimized)
    return F_optimized, global_inliers