import cv2
from EstimateFundamentalMatrix import plotEpipolarLines

def verifyWithOpenCV(F, x1, x2, image1, image2):
    img1 = image1.copy()
    img2 = image2.copy()
    F_cv, mask = cv2.findFundamentalMat(x1,x2,cv2.FM_LMEDS)

    plotEpipolarLines(F_cv, x1, x2, img1.copy(), img2.copy())
    plotEpipolarLines(F, x1, x2, img1.copy(), img2.copy())

    print(F)
    print(F_cv)
