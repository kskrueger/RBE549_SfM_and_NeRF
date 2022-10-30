# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EstimateFundamentalMatrix.py
import numpy as np

def estimateFundamentalMatrix(pt_correspondences):
    '''
    This function implements 8-pt algorithm to estimate the Fundamental matrix (F)
    input: get a set of feature point correspondences
    output: F matrix
    estimate the fundamental matrix using the 8-point algorithm
    x2_i^T . F . x1_i^T = 0

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
    # assert pt_correspondences.shape[0] == 8

    A = np.zeros((pt_correspondences.shape[0],9))
    
    def getRowOfA(row):
        x1,y1,x2,y2 = row[3:]
        return [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
    
    #create A matrix
    for i in range(pt_correspondences.shape[0]):
        A[i,:]=getRowOfA(pt_correspondences[i])

    # print(A)
    U,D,V = np.linalg.svd(A)
    F_vectorized = V[:,-1]
    F_noisy = F_vectorized.reshape(3,3)
    #SVD cleanup
    UF,UD,UV = np.linalg.svd(F_noisy)
    UD[-1]=0
    F = UF @ np.diag(UD) @ UV
    return F