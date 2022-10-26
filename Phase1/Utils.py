# import pandas

#default path
DEFAULT_MATCHES_FOLDER = './Data/P3Data/'
DEFAULT_CALIBRATION_FILENAME = './Data/P3Data/calibration.txt'

#read feature coresspondences
def read_matches(file_number, folder_path=DEFAULT_MATCHES_FOLDER):
    '''
    the function expects a .txt file
    parse the matching*.txt files to get the SIFT feature correspondences
    file_number belongs to [1,4]
    TODO: return a key-value dict of the feature correspondences
    '''
    file_path = folder_path+'matching'+str(file_number)+'.txt'
#     correspondences = []
#     with open(file_path, 'r'):
#         f.readlines()
    pass

#read K
def K_matrix(calib_file_path=DEFAULT_CALIBRATION_FILENAME):
    '''
    the function expects a .txt file
    parse the K matrix from the calibration file
    '''
    pass
