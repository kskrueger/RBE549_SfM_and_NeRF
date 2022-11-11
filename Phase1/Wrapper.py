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
from plotting import plotEpipolarLines
from plotting import plot_matches
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_path", default="./Data/UMD")
args = argparser.parse_args()

data_path = args.data_path
calibration_path = os.path.join(data_path, 'calibration.txt')

img_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
img_paths.extend(sorted(glob.glob(os.path.join(data_path, "*.jpg"))))
match_paths = glob.glob(os.path.join(data_path, "matching*.txt"))
calibration_file = open(calibration_path, 'r')

K = calibration_data = np.array([l.split(' ') for l in calibration_file], dtype=float)
imgs = [cv2.imread(img_path) for img_path in img_paths]
match_files = sorted([(int(m_path.split('matching')[-1].replace('.txt', '')), [l for l in open(m_path, 'r')]) for m_path in
               match_paths])

match_nfeatures = {}
match_data = {}
image_visbilities = {}
running_feature_idx_start = 0

for m_i, m_l in match_files:
    m_i = int(m_i)
    match_nfeatures[m_i] = int(m_l[0].split('nFeatures: ')[-1])
    features = {}
    visible_from_img_i = np.zeros((len(imgs), len(m_l[1:])))  # visbility matrix for IMG_I's matches for feature k in image J
    for feature_k, l in enumerate(m_l[1:]):
        # TODO: fix num_matches loop
        vals = np.array(l.strip(" \n").strip("\n").split(' '), dtype=float)
        num_matches, R, G, B, u_i, v_i, img_j, u_j, v_j = vals[:9]
        num_matches = int(num_matches)
        R = int(R)
        B = int(B)
        G = int(G)
        for k in range(num_matches-1):
            start = 6 + k*3
            img_j, u_j, v_j = vals[start:start+3]

            img_j = int(img_j)
            if img_j not in features.keys():
                features[img_j] = []
            features[img_j].append((num_matches, R, G, B, u_i, v_i, img_j, u_j, v_j, feature_k + running_feature_idx_start))
            running_feature_idx_start = feature_k
            visible_from_img_i[img_j-1, feature_k] = 1

    visible_from_img_i[m_i-1, :] = 1
    image_visbilities[m_i] = visible_from_img_i
    match_data[m_i] = features


def get_homogeneous(pts):
    return np.hstack((pts, np.ones((len(pts), 1))))


visibility_matrix = np.hstack([v for v in image_visbilities.values()])

for img_ii in match_data.keys():
    for img_jj in match_data[img_ii].keys():
        print("I, J", img_ii, img_jj)
        data = np.array(match_data[img_ii][img_jj])
        pts1 = data[:, 4:6]
        pts2 = data[:, 7:9]
        feature_indexes = data[:, -1].astype(int)


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

        ptsT = np.array([[1 / width, 0, -1],
                         [0, 1 / height, -1],
                         [0, 0, 1]])
        ptsT = np.linalg.inv(K)

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

        # TODO: try without normalized
        F2, F2_inliers = ransac_F(pts1n, pts2n, .001, 1000, imgs)
        # F2, F2_inliers = ransac_F(pts1, pts2, .005, 250, imgs)

        # F2, F2_inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, .3, 1, 250)

        # TODO: try without normalized
        F2un = np.dot(ptsT.T, np.dot(F2, ptsT))
        # F2un = F2
        F2un /= F2un[2, 2]
        F = F2un
        print("F", F)

        F2_inliers = F2_inliers[:, 0]  # .astype(bool)
        feature_indexes_i = feature_indexes[F2_inliers]
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
        E = get_E_from_F(K, F)
        print("E", E)

        # EE = K.T @ Fcv @ K

        cam_poses = get_camera_poses(E)

        cam_pose, lin_X, all_X = disambiguate_pose(None, cam_poses, pts1hi, pts2hi, K)
        from delete_helper import DisambiguateCameraPose

        # camX, camR, camC = DisambiguateCameraPose([c[0] for c in cam_poses], [c[1] for c in cam_poses], all_X)
        # lin_X = camX
        # cam_pose = (camC, camR)
        print("DisambiguateCameraPose", cam_pose[0])


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

        # plt.scatter(all_X[0][:, 0], all_X[0][:, 2], s=2, c='g')
        # plt.scatter(all_X[1][:, 0], all_X[1][:, 2], s=2, c='r')
        # plt.scatter(all_X[2][:, 0], all_X[2][:, 2], s=2, c='b')
        # plt.scatter(all_X[3][:, 0], all_X[3][:, 2], s=2, c='k')
        # plt.show()

        P0 = (np.zeros((3, 1)), np.eye(3))
        nonlin_X = nonlinear_triangulate(lin_X, P0, cam_pose, pts1i, pts2i, K)

        plt.scatter(lin_X[:, 0], lin_X[:, 2], s=2, c='r', label='Linear X')
        plt.scatter(nonlin_X[:, 0], nonlin_X[:, 2], s=2, c='g', label='NonLinear X')
        plt.legend()
        # plt.show()

        # from delete_helper import NonLinearTriangulation
        # nonTri = NonLinearTriangulation(K, pts1i, pts2i, pts_X_lin, np.eye(3), np.zeros((3, 1)), cam_poses[0][1], cam_poses[0][0])


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

        # from delete_helper import skew, compute_epipole, linear_triangulation_hartley
        # e = compute_epipole(F.T)  # left epipole
        # Te = skew(e)
        # pp = np.vstack((np.dot(Te, F.T).T, e)).T
        # lt = linear_triangulation_hartley(pts1hi.T, pts2hi.T, np.hstack((np.eye(3), np.zeros((3, 1)))), pp).T[:, :3]
        # plt.scatter(lt[:, 0], lt[:, 2])

        pnp_P, pnp_C, pnp_R, pnp_inliers = ransac_pnp(pts1i, nonlin_X, K, 25, 1000)

        feature_indexes_pnpi = feature_indexes_i[pnp_inliers]
        visibility_matrix[img_ii-1, feature_indexes_pnpi] = 2
        visibility_matrix[img_jj-1, feature_indexes_pnpi] = 2
        pts1i_pnpi = pts1i[pnp_inliers]
        X_pnpi = nonlin_X[pnp_inliers]

        pnp_nl_C, pnp_nl_R, pnp_nl_P = nonlinear_pnp(pnp_C, pnp_R, pts1i_pnpi, X_pnpi, K)

        eulers_lin = Rotation.from_matrix(pnp_R).as_euler('xyz')
        eulers = Rotation.from_matrix(pnp_nl_R).as_euler('xyz')

        plt.figure()
        plt.scatter(nonlin_X[:, 0], nonlin_X[:, 2], s=2, c='g', label='NonLinear X')

        plt.plot(pnp_C[0],
                 pnp_C[2],
                 marker=(3, 0, int(eulers_lin[1])),
                 markersize=15,
                 linestyle='None',
                 c='green',
                 label="Camera LinearPnP")

        plt.plot(pnp_nl_C[0],
                 pnp_nl_C[2],
                 marker=(3, 0, int(eulers[1])),
                 markersize=15,
                 linestyle='None',
                 c='purple',
                 label="Camera")
        plt.legend()
        # plt.show()

        # plt.figure()
        # plt.scatter(X_pnpi[:, 0], X_pnpi[:, 2], s=2)
        # plt.show()

        Xh_pnpi = np.hstack((X_pnpi, np.ones((len(X_pnpi), 1))))
        u1pnp = np.divide((np.dot(pnp_P[0, :], Xh_pnpi.T).T), (np.dot(pnp_P[2, :], Xh_pnpi.T).T))
        v1pnp = np.divide((np.dot(pnp_P[1, :], Xh_pnpi.T).T), (np.dot(pnp_P[2, :], Xh_pnpi.T).T))

        u1pnp_nl = np.divide((np.dot(pnp_nl_P[0, :], Xh_pnpi.T).T), (np.dot(pnp_nl_P[2, :], Xh_pnpi.T).T))
        v1pnp_nl = np.divide((np.dot(pnp_nl_P[1, :], Xh_pnpi.T).T), (np.dot(pnp_nl_P[2, :], Xh_pnpi.T).T))

        img_copy = imgs[0].copy()
        for u, v in pts1i_pnpi:
            cv2.circle(img_copy, (int(u), int(v)), 2, (0, 0, 255), -1)

        for u, v in zip(u1pnp, v1pnp):
            cv2.circle(img_copy, (int(u), int(v)), 3, (0, 255, 0), -1)

        for u, v in zip(u1pnp_nl, v1pnp_nl):
            cv2.circle(img_copy, (int(u), int(v)), 3, (255, 0, 255), -1)

        cv2.imshow("PnP Reprojections (Original=Red, LinearPnP=Green, NonLinearPnP=Purple)", img_copy)
        # cv2.waitKey(0)

        print("Finished, i, j", img_ii, img_jj)

# visibility =
# scipy.optimize.least_squares()

print("a")

# lns = plotEpipolarLines(F2un, pts1i, pts2i, imgs[0], imgs[1])
#
# from delete_helper import plot_epipolar_lines0, plot_epipolar_lines
#
# img1 = imgs[0].copy()
# img2 = imgs[1].copy()
#
# plot_epipolar_lines(pts1h.T, pts2h.T, F, True)
#
# print("a'")
# plot_epipolar_lines0(img1, img2, K, pts1h.T, pts2h.T, F2un)
#
# print('a')
