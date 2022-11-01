# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# EssentialMatrixFromFundamentalMatrix.py

def essentialMatrixFromFundamentalMatrix(F,K):
    '''
    Input 
        F = fundamental matrix
        K = Camera Intrinsics
    Output E
    E = K.T @ F @ K
    '''
    return K.T @ F @ K