# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# LinearTriangulation.py

def linearTriangulation():
    '''
    Inputs:
        uv_list      //List of all feature points (u,v) in n (=2) views (2,num_features,3,1)
        R_C_list,   //List of Camera pose for n (=2) views (2,(3,3),(3,1))
        K           //Camera intrinsics
    Outputs: 
        X = World coordinate (3,1)
    The feature point (u,v) should be the same as the reprojected world point
    use pinhole eqn to get the ideal uv ~ u_hat
    implement u X u_hat = 0 to get X
    all_X = []
    for uv in uv_list:
        UV = []
        P_i = K @ R_i|C_i (3,4)     //GET PROJECTION MATRIX
        UV.append(cross(uv_i,P_i))  //UV size = (3*n,4)
        U,D,V = svd(UV)
        X = V[:,-1][:-1]    //V size = (4,4)
        all_X.append(X)
    return X
    '''
    pass