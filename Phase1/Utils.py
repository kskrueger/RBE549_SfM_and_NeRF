import cv2
import numpy as np
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import plotEpipolarLines

def verifyWithOpenCV(F, x1, x2, image1, image2):
    img1 = image1.copy()
    img2 = image2.copy()
    F_cv, mask = cv2.findFundamentalMat(x1,x2,cv2.FM_LMEDS)

    plotEpipolarLines(F_cv, x1, x2, img1.copy(), img2.copy())
    plotEpipolarLines(F, x1, x2, img1.copy(), img2.copy())

    print(F)
    print(F_cv)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plotTriangulation(x,y,z=None,fig=None,is3D=False):
    N=30
    cmap = get_cmap(N)
    # if fig is None:
    #     fig = plt.figure()
    # if is3D:
    #     ax = fig.axes(projection='3d')
    #     ax.scatter3D(x, y, z, c=color,)
    # else:
    plt.scatter(x,y)
    plt.show()
    # return fig


def plotAllTriangulation(X_list, is3D=False):
    color_list=['r','g','b','k','o','p']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i,X in enumerate(X_list):
        if is3D:
            ax = fig.axes(projection='3d')
            ax.scatter3D(X[:,0],X[:,1],X[:,2])
        else:
            ax1.scatter(X[:,0],X[:,2],s=2,c=color_list[i])
    plt.show()
    # return fig