# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# DisambiguateCameraPose.py
import matplotlib.pyplot as plt
import numpy as np
import Utils
from LinearTriangulation import linearTriangulation

def disambiguateCamPoseAndTriangulate(pt_pair_list, pose_list, K):
    '''
    Check Chirality condition for all camera poses. Returns the (scaled) triangulated world coordinates along with the 
    corresponding camera pose (relative to the first camera) and number of points that satusfy the Chirality condition
    #### TODO update the docstring below
    Input:
        uv_list = list of point_pair correspondences between the two views (2,num_features,3,1)
        pose_list = List of camera poses (candidate camera poses for second camera)
        K = cam intrinsics (K is assumed to be same for all cameras i.e. same camera used for all pics)
    Output:
        triangulated X, Actual camera pose, inlier count
    C1,R1 = 0 matrix, I(3,3)    //considered at origin
    max_inliers = 0
    best_cam_pose = []
    for pose in pose_list:
        c2,r2 = pose
        X = linearTriangulation(uv_list, [[c1,r1],[c2,r2]], K)
        inliers = count(r2[3,;] @ (X-c2))
        if inliers > max_inliers:
            max_inliers = inliers
            best_cam_pose = pose
    return best_cam_pose // size = [(3,3),(3,1)]
    '''
    C1 = np.zeros(3)
    R1 = np.eye(3)
    max_inlier_count = 0
    best_cam_pose = []  #get T(or C) and R
    optimal_X = []
    candidate_X = []
    # fig,axes = 
    for pose in pose_list:
        C2, R2 = pose
        cam_pair_extrinsics = [[C1,R1],[C2,R2]]

        #get the estimated world corrdinates for the current extrinsics
        X = linearTriangulation(pt_pair_list, cam_pair_extrinsics, K)

        # Utils.plotTriangulation(X[:,0],X[:,2])
        candidate_X.append(X)
        
        #Check Chirality
        chirality_condition_check_vector = (R2[-1,:].T @ (X[:,:-1] - C2).T)

        inlier_count = (chirality_condition_check_vector>0).sum()
        
        if inlier_count > max_inlier_count:
            max_inliers = inlier_count
            best_cam_pose = pose
            optimal_X = X
        
        
    return optimal_X, best_cam_pose, candidate_X, max_inliers