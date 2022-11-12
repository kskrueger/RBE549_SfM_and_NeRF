# RBE549: Building Built in Minutes using NeRF
# Karter Krueger and Tript Sharma
# Loader.py

import glob
import json
import os
import numpy as np
import cv2
import torch


def LoadData(data_path, device='cpu'):
    print("Loading all data... ({})".format(data_path))
    train_f = open(os.path.join(data_path, "transforms_train.json"))
    test_f = open(os.path.join(data_path, "transforms_test.json"))
    val_f = open(os.path.join(data_path, "transforms_val.json"))

    train_json = json.load(train_f)
    test_json = json.load(test_f)
    val_json = json.load(val_f)

    # Load at half res
    new_sz = (200, 200)
    train_imgs = torch.from_numpy(np.array(
        [cv2.resize(cv2.imread(os.path.join(data_path, frame['file_path'][2:]) + ".png"), new_sz)/255.0 for frame in train_json['frames']]))
    test_imgs = torch.from_numpy(np.array(
        [cv2.resize(cv2.imread(os.path.join(data_path, frame['file_path'][2:]) + ".png"), new_sz)/255.0 for frame in test_json['frames']]))
    val_imgs = torch.from_numpy(np.array(
        [cv2.resize(cv2.imread(os.path.join(data_path, frame['file_path'][2:]) + ".png"), new_sz)/255.0 for frame in val_json['frames']]))

    H, W = train_imgs[0].shape[:2]
    x_FOV = train_json['camera_angle_x']
    focal_length = W / (2 * np.tan(x_FOV / 2))
    K = np.array([[focal_length, 0, W/2],
                  [0, focal_length, H/2],
                  [0, 0, 1]])

    train_T = torch.from_numpy(np.array([np.array(frame['transform_matrix']) for frame in train_json['frames']]))
    test_T = torch.from_numpy(np.array([np.array(frame['transform_matrix']) for frame in test_json['frames']]))
    val_T = torch.from_numpy(np.array([np.array(frame['transform_matrix']) for frame in val_json['frames']]))

    print("Done loading data")

    return K, train_imgs.to(device), train_T.to(device), test_imgs.to(device), \
           test_T.to(device), val_imgs.to(device), val_T.to(device)
