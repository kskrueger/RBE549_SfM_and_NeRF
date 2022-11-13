import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def estimateEpipole(F):
    '''
    Inputs:
        Fundamental Matrix F
    Outputs:
        Epipole

    Since epilines should pass through the epipole estimate epipole using the formula: F @ e = 0
    '''
    _,_,V = np.linalg.svd(F)
    e = V[-1]
    e /= e[-1]
    return e

def plotEpipolarLines(F, x1, x2, img1, img2):
    '''
    Input:
        F = Fundamental matrix
        x1 = image coordinates for img 1 (u,v)
        x1 = image coordinates for img 2 (u,v)
    F @ x1 = 0
    F.T @ x2 = 0
    '''
    x1 = np.hstack((x1, np.ones((x1.shape[0],1))))
    l1 = F @ x1.T   #x1 size = (num_features, 3)
    l1 /= l1[-1,:]

    x2 = np.hstack((x2, np.ones((x2.shape[0],1))))
    l2 = F.T @ x2.T
    # print(l2)
    l2 /= l2[-1,:]
    # print(l2)

    #get epipoles
    e1 = estimateEpipole(F)
    e2 = estimateEpipole(F.T)
    #draw lines
    # def drawEpilines(x, e):
    #     '''
    #     x = (3, num_features)
    #     e = epipole (3,1)
    #     '''
    #     m = (x-e)
    #     m = m[0,:]/m[1,:]

    for pt in x1:
        i1 = cv2.line(img1, pt[:-1].astype(np.int), e1[:-1].astype(np.int), (0,0,0), 2)
    for pt in x2:
        i2 = cv2.line(img2, pt[:-1].astype(np.int), e2[:-1].astype(np.int), (0,0,0), 2)

    cv2.imshow('1',i1)
    cv2.imshow('2',i2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def plot_matches(img1, img2, pts1, pts2, name="Matches"):
    img = cv2.drawMatches(img1,
                [cv2.KeyPoint(int(pt1[0]), int(pt1[1]), 1) for pt1 in pts1],
                img2,
                [cv2.KeyPoint(int(pt2[0]), int(pt2[1]), 1) for pt2 in pts2],
                [cv2.DMatch(i, i, 1) for i in np.arange(len(pts1))], None)
    cv2.imshow(name, img)
    cv2.waitKey(10)
    return img



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

def plot_camera(pnp_R, pnp_nl_R, nonlin_X, pnp_C, pnp_nl_C):
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
    plt.show()


