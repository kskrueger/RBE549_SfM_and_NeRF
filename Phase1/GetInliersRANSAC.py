# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# GetInliersRANSAC.py

def getInliersRANSAC():
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
    return optimised F
    '''
    pass