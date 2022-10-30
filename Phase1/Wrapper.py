# RBE549: Building Built in Minutes using SfM
# Karter Krueger and Tript Sharma
# Wrapper.py

import argparse
import numpy as np
import cv2
import os
from ParseData import *
from GetInliersRANSAC import getInliersRANSAC

argparser = argparse.ArgumentParser()
argparser.add_argument('--matches_folder_path', default=DEFAULT_MATCHES_FOLDER)
argparser.add_argument('--calib_file_path', default=DEFAULT_CALIBRATION_FILENAME)
args = argparser.parse_args()

K_matrix(args.calib_file_path)
corresponodences = parse_matches(file_number=1,folder_path=args.matches_folder_path)

img1 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/1.png'))
img2 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/2.png'))

getInliersRANSAC(corresponodences['12'], eps=20, n_iterations=100)