# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# LinearTriangulation.py
import numpy as np

def skew(x):
    '''
    Inputs: a vector
    Outputs: skew symmetric matrix
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def linearTriangulation(pt_pair_list, extrinsic_matrices, K):
    '''
    Function to find the world coordinates from a all image coordinate pairs (only for 2 views)
    Inputs:
        uv_list      //List of all feature points (u,v) in n (=2) views (2,num_features,3,1)
        R_C_list,   //List of Camera pose for n (=2) views (2,(3,3),(3,1))
        K           //Camera intrinsics
    Outputs: 
        X = World coordinate (3,1)
    The feature point (u,v) should be the same as the reprojected world point
    use pinhole eqn to get the ideal uv ~ u_hat
    implement np.cross(u, u_hat) = 0 i.e. u cross (P @ X) = 0 to get X
    all_X = []
    for uv in uv_list:
        UV = []
        P_i = K @ R_i|C_i (3,4)     //GET PROJECTION MATRIX
        UV.append(cross(uv_i,P_i))  //UV size = (3*n,4)
        U,D,V = svd(UV)
        X = last col of V.T    //V size = (4,4)
        all_X.append(X)
    return X
    '''
    X = []
    x1 = np.hstack((pt_pair_list[0],np.ones((pt_pair_list[0].shape[0],1))))
    x2 = np.hstack((pt_pair_list[1],np.ones((pt_pair_list[1].shape[0],1))))
    
    extrinsics1,extrinsics2 = extrinsic_matrices
    print(extrinsics1)
    C1, R1 = extrinsics1
    C2, R2 = extrinsics2
    
    P1 = K @ np.hstack((R1,C1.reshape((-1,1))))

    P2 = K @ np.hstack((R2,C2.reshape((-1,1))))

    for x1i,x2i in zip(x1,x2):
        x1i_skew = skew(x1i)
        x2i_skew = skew(x2i)
        
        AX = [[x1i_skew @ P1],[x2i_skew @ P2]]
        AX = np.array(AX).reshape((-1,4))
        _,_,V = np.linalg.svd(AX)

        Xi = V[-1,:]
        Xi /= Xi[-1]
        X.append(Xi)
    print(X)
    return np.array(X)