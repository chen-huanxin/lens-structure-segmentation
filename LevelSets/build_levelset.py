from skimage import segmentation
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def EuclideanDistance(x1,y1,x2,y2):
    return np.sqrt(np.power(np.abs(x1 - x2), 2) + np.power(np.abs(y1 - y2), 2))

def ManhattanDistance(x1,y1,x2,y2):
    return np.abs(x1 - x2)+np.abs(y1 - y2)

def Chebyshevdistance(x1,y1,x2,y2):
    a = np.abs(x1 - x2)
    b = np.abs(y1 - y2)
    if a > b:
        return a
    else:
        return b

def compute_distance(new_idx, x, y, option=1):
    min_distance = None
    for boudry_y, boudry_x in new_idx:
        if option == 1:
            distance = EuclideanDistance(x1=boudry_x, y1=boudry_y, x2=x,y2=y)
        if option == 2:
            distance = ManhattanDistance(x1=boudry_x, y1=boudry_y, x2=x, y2=y)
        if option == 3:
            distance = Chebyshevdistance(x1=boudry_x, y1=boudry_y, x2=x, y2=y)
        if min_distance is None or distance < min_distance:
            min_distance = distance
    return min_distance

def compute_graph(img,options=1):
    boudry = segmentation.find_boundaries(img)
    boudry = boudry * 1
    all_y, all_x = np.where(boudry == 1)
    ans_matrx = np.ones_like(boudry)
    ans_matrx[img != 1] = -1
    new_idx = zip(all_y, all_x)
    y_axis, x_axis = ans_matrx.shape
    for y in range(y_axis):
        for x in range(x_axis):
            ans_matrx[y, x] = compute_distance(new_idx, x, y,option=options)
    return ans_matrx

data_path = '/home/intern1/yinpengshuai/levelset/data/LevelSetDataSet/train_label/'
label_list = os.listdir(data_path)
# img_list = ['1390_L_004_110345.png']
save_path = '/home/intern1/yinpengshuai/levelset/data/LevelSetDataSet/Level_set_label/'
showLevelSet = 0

for idx,label_name in enumerate(label_list):
    path = os.path.join(data_path, label_name)
    savedPath_name = os.path.join(save_path, label_name)
    img = cv2.imread(path,0)
    #img = cv2.resize(img, (64, 64))

    #Nucleus Area
    NucleusArea = (img==1)
    CortexArea  = (img==0)

    InversLabel = np.zeros_like(img)
    InversLabel[CortexArea] = 1
    InversLabel[NucleusArea] = 0

    #Find boundary
    #boundary = segmentation.find_boundaries(img,mode='inner')
    #boundary = boundary * 1
    #boundary = boundary.astype(np.uint8)
    #img = cv2.resize(img,(128,128))
    TransformedImgTopZero = cv2.distanceTransform(img, cv2.cv.CV_DIST_L2, 5)
    TransformedImgDownZero = cv2.distanceTransform(InversLabel,cv2.cv.CV_DIST_L2,5)

    Final = TransformedImgTopZero - TransformedImgDownZero
    np.save(savedPath_name,Final)

    #Plot the graph
    if showLevelSet==1:
        fig = plt.figure()
        ax = Axes3D(fig)
        # X, Y value
        X = np.arange(0, Final.shape[1], 1)
        Y = np.arange(0, Final.shape[0], 1)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        ax.plot_surface(X, Y, Final, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))


