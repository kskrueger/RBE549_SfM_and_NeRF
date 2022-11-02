# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EssentialMatrixFromFundamentalMatrix.py
import numpy as np

def essentialMatrixFromFundamentalMatrix(F,K):
    '''
    Input 
        F = fundamental matrix
        K = Camera Intrinsics
    Output E
    E = K.T @ F @ K
    
    '''
    E =  K.T @ F @ K
    #cleanup E
    U,D,VT = np.linalg.svd(E)
    E_hat = U @ np.diag([1,1,0]) @ VT
    return E_hat
