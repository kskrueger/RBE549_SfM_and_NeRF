# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# Wrapper.py

import argparse
import numpy as np
import cv2
import os
from ParseData import *
from GetInliersRANSAC import getInliersRANSAC
from EstimateFundamentalMatrix import plotEpipolarLines

argparser = argparse.ArgumentParser()
argparser.add_argument('--matches_folder_path', default=DEFAULT_MATCHES_FOLDER)
argparser.add_argument('--calib_file_path', default=DEFAULT_CALIBRATION_FILENAME)
args = argparser.parse_args()

K_matrix(args.calib_file_path)
correspondences = parse_matches(file_number=1,folder_path=args.matches_folder_path)

img1 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/1.png'))
img2 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/2.png'))

F, inlier_set  = getInliersRANSAC(correspondences['12'], eps=40, n_iterations=150)
print(F)
# show_feature_matches(img1,img2,correspondences['12'][inlier_set,:])

plotEpipolarLines(F, correspondences['12'][inlier_set,3:5],correspondences['12'][inlier_set,5:7], img1, img2)
