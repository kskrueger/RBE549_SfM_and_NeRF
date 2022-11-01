# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# ExtractCameraPose.py
import numpy as np

def CamPoseFromE(E):
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
    pose_list = []

    U,D,V = np.linalg.svd(E)
    C = [U[:,-1], -U[:,-1]]
    W = np.array([  [0,1,0],
                    [-1,0,0],
                    [0,0,1]
                ])
    R = [U@W@V, U@W.T@V]

    for i in range(len(R)):
        if np.linalg.det(R[i])>0:
            pose_list.append([C[0], R[i]])
            pose_list.append([C[1], R[i]])
        else:
            pose_list.append([-C[0], -R[i]])
            pose_list.append([-C[1], -R[i]])
    print(pose_list)
    return pose_list