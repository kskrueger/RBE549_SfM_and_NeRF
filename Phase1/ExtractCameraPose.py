# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# ExtractCameraPose.py

def CamPoseFromE():
    '''
    Inputs: 
        E = essential matrix
    Output:
        list of possible camera T and R
    returns possible camera pose configurations from E
    U,D,V.T = svd(E)
    T or C = +- U[:,-1]
    R = U @ W @ V.T or U @ W.T @ V.T
    this gives us 4 combinations of C and R
    if det(R)<0
        T = -T      
        R = -R
    because we assume Right Hand Coordinate System so det(R) should be 1
    '''
    pass