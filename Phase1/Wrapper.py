# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# Wrapper.py

import glob
import os
import cv2
import numpy as np
import argparse
from EstimateFundamentalMatrix import get_fundamental_mat
from EssentialMatrixFromFundamentalMatrix import get_E_from_F
from GetInliersRANSAC import ransac_F
from LinearTriangulation import linear_triangulate
from NonlinearTriangulation import nonlinear_triangulate
from DisambiguateCameraPose import disambiguate_pose
from ExtractCameraPose import get_camera_poses
from plotting import plotEpipolarLines
from plotting import plot_matches

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_path", default="./Data/P3Data")
args = argparser.parse_args()

data_path = args.data_path
calibration_path = os.path.join(data_path, 'calibration.txt')

img_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
match_paths = glob.glob(os.path.join(data_path, "matching*.txt"))
calibration_file = open(calibration_path, 'r')

K = calibration_data = np.array([l.split(' ') for l in calibration_file], dtype=float)
imgs = [cv2.imread(img_path) for img_path in img_paths]
match_files = [(int(m_path.split('matching')[-1].replace('.txt', '')), [l for l in open(m_path, 'r')]) for m_path in match_paths]

match_nfeatures = {}
match_data = {}

for m_i, m_l in match_files:
    match_nfeatures[int(m_i)] = int(m_l[0].split('nFeatures: ')[-1])
    features = {}
    for l in m_l[1:]:
        num_matches, R, G, B, u_i, v_i, i, u_j, v_j = np.array(l.strip("\n").split(' ')[:9], dtype=float)
        num_matches = int(num_matches)
        R = int(R)
        B = int(B)
        G = int(G)
        i = int(i)
        if i not in features.keys():
            features[i] = []
        features[i].append((num_matches, R, G, B, u_i, v_i, i, u_j, v_j))

    match_data[m_i] = features

def get_homogeneous(pts):
    return np.hstack((pts, np.ones((len(pts), 1))))

data = np.array(match_data[1][2])
pts1 = data[:, 4:6]
pts2 = data[:, 7:9]

plot_matches(imgs[0], imgs[1], pts1, pts2)

# img_gray = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(img_gray, None)

# a = cv2.goodFeaturesToTrack(img_gray, 1000, .01, 5)
# img_out = imgs[0].copy()
# for i in range(len(a)):
#     cv2.circle(img_out, a[i, 0, :].astype(int), 5, (0, 255, 0), -1)
# cv2.imshow("img_out", img_out)
# cv2.waitKey(0)

# img = cv2.drawKeypoints(img_gray,kp,imgs[0])
# cv2.imshow("Sift", img)
# cv2.waitKey(0)



width, height = imgs[0].shape[:2]
pts1h = get_homogeneous(pts1)
pts2h = get_homogeneous(pts2)

ptsT = np.array([[1/width, 0, -1],
                 [0, 1/height, -1],
                 [0, 0, 1]])
K_inv = np.linalg.inv(calibration_data)
pts1nh = (ptsT @ pts1h.T).T
pts2nh = (ptsT @ pts2h.T).T
pts1n = pts1nh[:, :2]
pts2n = pts2nh[:, :2]

# Fcv, Fcv_inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, .3, 1, 100)

pts1nh = get_homogeneous(pts1n)
pts2nh = get_homogeneous(pts2n)

# inlier_errors = np.zeros((len(pts1nh), 1))
# for j in range(len(pts1nh)):
#     inlier_errors[j] = pts2nh[j].T @ (Fcv @ pts1nh[j])
# inliers_idx = np.abs(inlier_errors) < .3

F2, F2_inliers = ransac_F(pts1n, pts2n, .05, 250, imgs)

# F2, F2_inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, .3, 1, 250)

F2un = np.dot(ptsT.T, np.dot(F2, ptsT))
F2un /= F2un[2, 2]
F = F2un
print("F", F)

F2_inliers = F2_inliers[:, 0]#.astype(bool)
pts1i = pts1[F2_inliers]
pts2i = pts2[F2_inliers]
pts1hi = pts1h[F2_inliers]
pts2hi = pts2h[F2_inliers]
pts1nhi = pts1nh[F2_inliers]
pts2nhi = pts2nh[F2_inliers]
pts1ni = pts1n[F2_inliers]
pts2ni = pts2n[F2_inliers]

plot_matches(imgs[0], imgs[1], pts1i, pts2i)

# E = get_E_from_F(K, F2)
E = get_E_from_F(K, F2un)
print("E", E)

# EE = K.T @ Fcv @ K

cam_poses = get_camera_poses(E)

cam_pose, pts_X_lin, all_X = disambiguate_pose(None, cam_poses, pts1hi, pts2hi, K)
plt.scatter(pts_X_lin[:, 0], pts_X_lin[:, 2], s=2, c='g')
plt.show()

def plot_3d(X):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 0], X[:, 2], X[:, 1])

    ax.set_xlim(-6, 8)
    ax.set_ylim(2, 18)
    ax.set_zlim(-10, 10)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    plt.show()

# plot_3d(all_X[0])
# plot_3d(all_X[1])
# plot_3d(all_X[2])
# plot_3d(all_X[3])

plt.scatter(all_X[0][:, 0], all_X[0][:, 2], s=2, c='g')
plt.scatter(all_X[1][:, 0], all_X[1][:, 2], s=2, c='r')
plt.scatter(all_X[2][:, 0], all_X[2][:, 2], s=2, c='b')
plt.scatter(all_X[3][:, 0], all_X[3][:, 2], s=2, c='k')
plt.show()

P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
nonlinTri = nonlinear_triangulate(pts_X_lin, P0, cam_pose, pts1i, pts2i)

# from delete_helper import skew, compute_epipole, linear_triangulation_hartley
# e = compute_epipole(F.T)  # left epipole
# Te = skew(e)
# pp = np.vstack((np.dot(Te, F.T).T, e)).T
# lt = linear_triangulation_hartley(pts1hi.T, pts2hi.T, np.hstack((np.eye(3), np.zeros((3, 1)))), pp).T[:, :3]
# plt.scatter(lt[:, 0], lt[:, 2])

print("a")

lns = plotEpipolarLines(F2un, pts1i, pts2i, imgs[0], imgs[1])

from delete_helper import plot_epipolar_lines0, plot_epipolar_lines


img1 = imgs[0].copy()
img2 = imgs[1].copy()

plot_epipolar_lines(pts1h.T, pts2h.T, F, True)

print("a'")
plot_epipolar_lines0(img1, img2, K, pts1h.T, pts2h.T, F2un)

print('a')
