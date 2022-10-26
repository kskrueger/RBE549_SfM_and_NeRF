# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EstimateFundamentalMatrix.py

def estimateFundamentalMatrix():
    '''
    input: get the set of 8 feature correspondences
    outout: F matrix
    estimate the fundamental matrix using the 8-point algorithm
    x2_i^T . F . x1_i^T = 0
    select 8 random point pairs
    create A matrix as follows
        A = [x1_1.x2_1 y1_1.x2_1 x2_1 x_1^1.y2_1 y1_1.y2_1 y2_1 x1_1 y1_1 1]
    A.f = 0
    f = last col vector of V of svd(A)
    reshape f to (3,3)

    (do SVD cleanup to get constrain rank to 2 as follows)
    svd(F)
    D[3,3]=0
    U*D*V^T = F (cleanup)
    '''
    pass
