# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# DisambiguateCameraPose.py

def disambiguateCamPose():
    '''
    Check Chirality condition for all camera poses
    Input:
        uv_list = point_pair correspondences between the two views (2,num_features,3,1)
        pose_list = List of camera poses (candidate camera poses for second camera)
        K = cam intrinsics (K is assumed to be same for all cameras i.e. same camera used for all pics)
    Output:
        Actual camera pose
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