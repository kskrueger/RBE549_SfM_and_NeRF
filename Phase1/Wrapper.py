# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# Wrapper.py

import glob
import os
import cv2
import numpy as np
import argparse

import scipy.optimize

from EstimateFundamentalMatrix import get_fundamental_mat
from EssentialMatrixFromFundamentalMatrix import get_E_from_F
from GetInliersRANSAC import ransac_F
from LinearTriangulation import linear_triangulate
from NonlinearPnP import nonlinear_pnp
from NonlinearTriangulation import nonlinear_triangulate
from DisambiguateCameraPose import disambiguate_pose
from PnPRANSAC import ransac_pnp
from ExtractCameraPose import get_camera_poses
from plotting import plotEpipolarLines, plot_camera
from plotting import plot_matches
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_path", default="./Data/P3Data")
args = argparser.parse_args()

data_path = args.data_path
calibration_path = os.path.join(data_path, 'calibration.txt')

img_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
img_paths.extend(sorted(glob.glob(os.path.join(data_path, "*.jpg"))))
match_paths = glob.glob(os.path.join(data_path, "matching*.txt"))
calibration_file = open(calibration_path, 'r')

K = calibration_data = np.array([l.split(' ') for l in calibration_file], dtype=float)
imgs = [cv2.imread(img_path) for img_path in sorted(img_paths)]
match_files = sorted([(int(m_path.split('matching')[-1].replace('.txt', '')), [l for l in open(m_path, 'r')]) for m_path in
               match_paths])

match_nfeatures = {}
match_data = {}
image_visbilities = {}
running_feature_idx_start = 0

file_start = 0
all_m_features = []
visible_lines = []
pts_lines = []
for m_i, m_l in match_files:
    m_i = int(m_i)
    match_nfeatures[m_i] = int(m_l[0].split('nFeatures: ')[-1])
    features = {}
    visible_from_img_i = np.zeros((len(imgs), len(m_l[1:])))  # visbility matrix for IMG_I's matches for feature k in image J
    for feature_k, l in enumerate(m_l[1:]):
        vals = np.array(l.strip(" \n").strip("\n").split(' '), dtype=float)
        num_matches, R, G, B, u_i, v_i, img_j, u_j, v_j = vals[:9]
        num_matches = int(num_matches)
        R = int(R)
        B = int(B)
        G = int(G)
        visible_line = np.zeros((len(imgs)), dtype=int)
        visible_line[m_i-1] = 1
        pts_line = np.zeros((len(imgs), 2), dtype=float)
        pts_line[m_i-1, :] = u_i, v_i
        for k in range(num_matches-1):
            start = 6 + k*3
            img_j, u_j, v_j = vals[start:start+3]

            img_j = int(img_j)
            if img_j not in features.keys():
                features[img_j] = []
            features[img_j].append((m_i, R, G, B, u_i, v_i, img_j, u_j, v_j, feature_k + running_feature_idx_start))
            all_m_features.append((m_i, R, G, B, u_i, v_i, img_j, u_j, v_j, feature_k + running_feature_idx_start + file_start))
            visible_from_img_i[img_j-1, feature_k] = 1
            visible_line[img_j - 1] = 1
            pts_line[img_j - 1, :] = u_j, v_j
        running_feature_idx_start = feature_k
        visible_lines.append(visible_line)
        pts_lines.append(pts_line)

    visible_from_img_i[m_i-1, :] = 1
    image_visbilities[m_i] = visible_from_img_i
    match_data[m_i] = features
    file_start = running_feature_idx_start

all_data = np.array(all_m_features)
new_visible = np.stack(visible_lines).astype(int)
new_x = np.stack(pts_lines).astype(float)

def get_homogeneous(pts):
    return np.hstack((pts, np.ones((len(pts), 1))))

# visibility_matrix0 = np.hstack([v for v in image_visbilities.values()])
# visibility_matrix = np.zeros_like(visibility_matrix0)

# Step 1: Prune image features (already done before reading in the files)

# for img_ii in match_data.keys():
    # for img_jj in match_data[img_ii].keys():

# Step 2: Compute F and E for features between images 1 and 2
img_ii = 1
img_jj = 4

allC = []
allR = []
allX = np.zeros((len(new_x), 3))
# new_visible = np.zeros((len(m), len(imgs)))
camOrder = [img_ii, img_jj]
good_idx = np.zeros((len(new_x), 1), dtype=int)

# all_match_data = np.vstack([np.vstack([v for v in match_data[img_k].values()]) for img_k in match_data.keys()])
# all_img1_idx = m[:, 0].astype(int)
# all_img2_idx = m[:, 6].astype(int)

inliers_ij = np.where((new_visible[:, img_ii-1]) & (new_visible[:, img_jj-1]))[0]

print("I, J", img_ii)
# data = np.array(match_data[img_ii][img_jj])
# pts1 = data[:, 4:6]
# pts2 = data[:, 7:9]
pts1 = new_x[inliers_ij, img_ii-1]
pts2 = new_x[inliers_ij, img_jj-1]
feature_indexes = inliers_ij.astype(int) #data[:, -1].astype(int)

plot_matches(imgs[img_ii-1], imgs[img_jj-1], pts1, pts2)

width, height = imgs[0].shape[:2]
pts1h = get_homogeneous(pts1)
pts2h = get_homogeneous(pts2)

# ptsT = np.array([[1 / width, 0, -1],
#                  [0, 1 / height, -1],
#                  [0, 0, 1]])
ptsT = np.linalg.inv(K)

K_inv = np.linalg.inv(calibration_data)
pts1nh = (ptsT @ pts1h.T).T
pts2nh = (ptsT @ pts2h.T).T
pts1n = pts1nh[:, :2]
pts2n = pts2nh[:, :2]

pts1nh = get_homogeneous(pts1n)
pts2nh = get_homogeneous(pts2n)

F2, F2_inliers = ransac_F(pts1n, pts2n, .25, 5000, imgs)

F2un = np.dot(ptsT.T, np.dot(F2, ptsT))
F2un /= F2un[2, 2]
F = F2un
print("F", F)

F2_inliers = F2_inliers[:, 0]
feature_indexes_i = feature_indexes[F2_inliers]
good_idx[feature_indexes_i] = 1
pts1i = pts1[F2_inliers]
pts2i = pts2[F2_inliers]
pts1hi = pts1h[F2_inliers]
pts2hi = pts2h[F2_inliers]
pts1nhi = pts1nh[F2_inliers]
pts2nhi = pts2nh[F2_inliers]
pts1ni = pts1n[F2_inliers]
pts2ni = pts2n[F2_inliers]

plot_matches(imgs[img_ii-1], imgs[img_jj-1], pts1i, pts2i, "Matches F_Inlier")

E = get_E_from_F(K, F)
print("E", E)

cam_poses = get_camera_poses(E)

cam_pose, lin_X, all_X = disambiguate_pose(None, cam_poses, pts1hi, pts2hi, K)
print("DisambiguateCameraPose", cam_pose[0])
allC.append(cam_pose[0])
allR.append(cam_pose[1])

# plot_3d(all_X[0])
# plot_3d(all_X[1])
# plot_3d(all_X[2])
# plot_3d(all_X[3])

# plt.scatter(all_X[0][:, 0], all_X[0][:, 2], s=2, c='g')
# plt.scatter(all_X[1][:, 0], all_X[1][:, 2], s=2, c='r')
# plt.scatter(all_X[2][:, 0], all_X[2][:, 2], s=2, c='b')
# plt.scatter(all_X[3][:, 0], all_X[3][:, 2], s=2, c='k')
# plt.show()

P0 = (np.zeros((3, 1)), np.eye(3))
nonlin_X = nonlinear_triangulate(lin_X, P0, cam_pose, pts1i, pts2i, K)
allX[good_idx[:, 0] == 1] = nonlin_X

# plt.scatter(lin_X[:, 0], lin_X[:, 2], s=2, c='r', label='Linear X')
# plt.scatter(nonlin_X[:, 0], nonlin_X[:, 2], s=2, c='g', label='NonLinear X')
# plt.legend()
# plt.show()

P1 = np.dot(K, np.dot(P0[1], np.hstack((np.eye(3), -P0[0]))))
P2 = np.dot(K, np.dot(cam_poses[0][1], np.hstack((np.eye(3), -cam_poses[0][0]))))

X = np.hstack((nonlin_X, np.ones((len(nonlin_X), 1))))
u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))

img_copy = imgs[0].copy()
for u, v in zip(u1, v1):
    cv2.circle(img_copy, (int(u), int(v)), 3, (0, 255, 0), -1)

for u, v in pts1i:
    cv2.circle(img_copy, (int(u), int(v)), 2, (0, 0, 255), -1)

cv2.imshow("NonLinearX Reprojection", img_copy)
# cv2.waitKey(0)

plt.figure()
plt.scatter(nonlin_X[:, 0], nonlin_X[:, 2], s=2, c='g', label='NonLinear X')
colors = ['b', 'r', 'k', 'purple', 'pink', 'orange']

eulers = Rotation.from_matrix(allR[0]).as_euler('xyz')
plt.plot(allC[0][0],
         allC[0][2],
         marker=(3, 0, int(eulers[1])),
         markersize=15,
         linestyle='None',
         c=colors[img_ii],
         label="Camera ".format(img_ii))

# PnP applied to indexes 3-I
for img_jj in match_data[img_ii].keys():
    if img_jj in camOrder:
        print("Already processed cam")
        continue
    # data = np.array(match_data[img_ii][img_jj])
    inliers_ij2 = np.where((new_visible[:, img_ii-1]) & (new_visible[:, img_jj-1]) & good_idx[:, 0])[0]

    print("I, J", img_ii)
    pts_i = new_x[inliers_ij2, img_ii-1]
    pts_j = new_x[inliers_ij2, img_jj-1]
    feature_indexes_j = inliers_ij2.astype(int)
    x_inliers = allX[feature_indexes_j]
    pnp_P, pnp_C, pnp_R, pnp_inliers = ransac_pnp(pts_j, x_inliers, K, 50, 1000)

    feature_indexes_pnp_j = feature_indexes_j[pnp_inliers]
    # visibility_matrix[img_ii-1, feature_indexes_pnp_j] = 1
    # visibility_matrix[img_jj-1, feature_indexes_pnp_j] = 1
    pts_j_pnpi = pts_j[pnp_inliers]
    X_pnpi = x_inliers[pnp_inliers]

    pnp_nl_C, pnp_nl_R, pnp_nl_P = nonlinear_pnp(pnp_C, pnp_R, pts_j_pnpi, X_pnpi, K)
    camOrder.append(img_jj)
    allR.append(pnp_nl_R)
    allC.append(pnp_nl_C)

    eulers = Rotation.from_matrix(pnp_nl_R).as_euler('xyz')
    plt.plot(pnp_nl_C[0],
             pnp_nl_C[2],
             marker=(3, 0, int(eulers[1])),
             markersize=15,
             linestyle='None',
             c=colors[img_jj],
             label="Camera {}".format(img_jj))

    # plot_camera(pnp_R, pnp_nl_R, nonlin_X, pnp_C, pnp_nl_C)

    # plt.figure()
    # plt.scatter(X_pnpi[:, 0], X_pnpi[:, 2], s=2)
    # plt.show()

    Xh_pnpi = np.hstack((X_pnpi, np.ones((len(X_pnpi), 1))))
    u1pnp = np.divide((np.dot(pnp_P[0, :], Xh_pnpi.T).T), (np.dot(pnp_P[2, :], Xh_pnpi.T).T))
    v1pnp = np.divide((np.dot(pnp_P[1, :], Xh_pnpi.T).T), (np.dot(pnp_P[2, :], Xh_pnpi.T).T))

    u1pnp_nl = np.divide((np.dot(pnp_nl_P[0, :], Xh_pnpi.T).T), (np.dot(pnp_nl_P[2, :], Xh_pnpi.T).T))
    v1pnp_nl = np.divide((np.dot(pnp_nl_P[1, :], Xh_pnpi.T).T), (np.dot(pnp_nl_P[2, :], Xh_pnpi.T).T))

    img_copy = imgs[img_ii].copy()
    for u, v in pts_j_pnpi:
        cv2.circle(img_copy, (int(u), int(v)), 2, (0, 0, 255), -1)

    for u, v in zip(u1pnp, v1pnp):
        cv2.circle(img_copy, (int(u), int(v)), 3, (0, 255, 0), -1)

    for u, v in zip(u1pnp_nl, v1pnp_nl):
        cv2.circle(img_copy, (int(u), int(v)), 3, (255, 0, 255), -1)

    cv2.imshow("PnP Reprojections on IMG {}-{} : (Original=Red, LinearPnP=Green, NonLinearPnP=Purple)".format(img_ii, img_jj), img_copy)
    # cv2.waitKey(0)

    print("Finished, i, j", img_ii, img_jj)


plt.legend()
plt.plot()

# Run bundle adjustment here and re-plot the final camera poses and point positions

print("Done")