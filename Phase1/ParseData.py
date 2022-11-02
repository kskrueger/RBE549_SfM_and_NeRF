# import pandas
import argparse
import os
import glob
import numpy as np
import cv2

#default path
DEFAULT_MATCHES_FOLDER = 'Phase1/Data/P3Data/'
DEFAULT_CALIBRATION_FILENAME = 'Phase1/Data/P3Data/calibration.txt'


#read feature coresspondences
def parse_matches(file_number, folder_path=DEFAULT_MATCHES_FOLDER):
    '''
    the function expects a .txt file
    parse the matching*.txt files to get the SIFT feature correspondences
    file_number belongs to [1,4]
    
    # Input:
    #   Each Row: (num_matches) R G B (ucurrent image) (vcurrent image) (image id) (uimage id image) (vimage id image) ... for num_matches
    #Returns:
    #   'img1_id concatenated with img2_id': [[R G B (ucurrent image) (vcurrent image) (image id) (uimage id image) (vimage id image)]...]
    '''
    data_path = 'matching'+str(file_number)+'.txt'
    file_path = os.path.join(os.getcwd(), folder_path, data_path)
    correspondences = []
    #create the list of correspondences (list of str)
    with open(file_path, 'r') as f:
        correspondences = f.readlines()
    #process the list of correspondences (list of list of str)    
    correspondences = [x.split() for x in correspondences[1:]]  #first line contains unuseful info

    #convert list of list of str to dict of array
    matches_dict = {}
    for x in correspondences:
        correspondence = x[1:6] # RGB value, current (u,v)

        #separate each feature correspondence and add em to respective dict key
        for i in range(1,int(x[0])):
            current_correspondence = correspondence.copy()
            current_correspondence.extend([x[-(i*3)+1],x[-(i*3)+2]])

            key = str(file_number)+x[-(i*3)]
            if key in matches_dict:
                matches_dict[key].append(current_correspondence)
            else:
                matches_dict[key] = [current_correspondence]
    #conversion to array
    for key in matches_dict:
        matches_dict[key] = np.array(matches_dict[key], dtype=np.float32)
    print(matches_dict)
    return matches_dict

#read K
def K_matrix(calib_file_path=DEFAULT_CALIBRATION_FILENAME):
    '''
    the function expects relative path to a .txt file
    parse the K matrix from the calibration file
    '''
    K = []
    #read file
    with open(calib_file_path, 'r') as f:
        K = f.readlines()
    #create K matrix and do type casting
    K = np.array([x.split() for x in K], dtype=np.float32)
    # print(K)
    return K

def show_feature_matches(img1, img2, correspondences):
    #create image
    matches_img = np.hstack([img1, img2])
    #draw matches
    matches_img = cv2.drawMatches(img1, [cv2.KeyPoint(x[3], x[4], 3) for x in correspondences],
							  img2, [cv2.KeyPoint(x[5], x[6], 3) for x in correspondences],
							  [cv2.DMatch(index, index, 0) for index in range(correspondences.shape[0])], matches_img, (0, 255, 0), (0, 0, 255))
		
    cv2.imshow('feature_matches',matches_img)
    # cv2.imwrite('feature_matches.jpg',matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--matches_folder_path', default=DEFAULT_MATCHES_FOLDER)
    argparser.add_argument('--calib_file_path', default=DEFAULT_CALIBRATION_FILENAME)
    args = argparser.parse_args()

    # K_matrix(args.calib_file_path)
    corresponodences = parse_matches(file_number=1,folder_path=args.matches_folder_path)

    img1 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/1.png'))
    img2 = cv2.imread(os.path.join(os.getcwd(),'Phase1/Data/P3Data/2.png'))
    show_feature_matches(img1,img2,corresponodences['12'])
        
