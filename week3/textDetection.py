import matplotlib.pyplot as plt
from utils import *
import cv2
import pickle
import ml_metrics as metrics
import numpy as np


# It computes our method to find the bounding box of the text box
def calculate_txtbox(image,gtBbox):
    plots = 0
    kernel = np.ones((30, 30), np.uint8)
    k = np.ones((15,15),np.uint8)
    kernel_ = np.ones((8,round(np.size(image,1)/8)))
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # s = hsv[:,:,1]
    image = np.float32(image)
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image,(5,5))
    t = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,kernel)
    b = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel_)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel_)
    # t = cv2.dilate(t, kernel, iterations=1)
    # b = cv2.dilate(b, kernel, iterations=1)
    retval, t = cv2.threshold(t,round(np.max(t)*0.65),255,cv2.THRESH_BINARY)
    retval, b = cv2.threshold(t,round(np.max(b)*0.65),255,cv2.THRESH_BINARY)
    t = cv2.dilate(t, k, iterations=1)
    b = cv2.dilate(b, k, iterations=1)
    im1 = t.astype(np.uint8)
    im2 = b.astype(np.uint8)
    # Compute point contours
    contours1, thr = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, thr = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Create bboxes
    x1, y1, w1, h1 = filterContours(contours1, image, im1)
    x2, y2, w2, h2 = filterContours(contours2, image, im2)
    if plots:
        hola = cv2.drawContours(image, contours1, -1, (0, 127, 0), 8)
        plt.imshow(t, 'gray'), plt.show()
        plt.imshow(b, 'gray'), plt.show()
        cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), 255, 3)
        plt.imshow(image),plt.show()
        cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), 200, 3)
        plt.imshow(image),plt.show()
    X = [x1,x2]
    Y = [y1,y2]
    W = [w1,w2]
    H = [h1,h2]
    ind = chooseBestBbox(W, H)
    iouSingle = iou([X[ind], Y[ind], (X[ind] + W[ind]), (Y[ind] + H[ind])], gtBbox)

    return iouSingle, [X[ind], Y[ind], W[ind], H[ind]]
