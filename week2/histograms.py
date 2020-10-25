import cv2
import numpy as np


def rgbHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        im = listIm[i]
        b = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
        g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
        r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
        concat = np.concatenate([b, g, r], axis=0)
        concat = concat / np.max(concat)
        concat = np.array(concat, dtype=np.float32)
        featVecs.append(concat)
    return featVecs

def grayHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        # cv2.imshow('og', img)
        # cv2.waitKey()
        imgGray = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', imgGray)
        # cv2.waitKey()
        histr = cv2.calcHist([imgGray], [0], None, [nBins], [0, 255])
        histr = histr/np.max(histr)
        histr = np.array(histr, dtype=np.float32)
        featVecs.append(histr)
    return featVecs


def labHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        hsv = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2LAB)
        l = np.histogram(hsv[:, :, 0], nBins, [0, 255])[0]
        a = np.histogram(hsv[:, :, 1], nBins, [0, 255])[0]
        b = np.histogram(hsv[:, :, 2], nBins, [0, 255])[0]
        concat = np.concatenate([l, a, b], axis=0)
        concat = concat/np.max(concat)
        concat = np.array(concat, dtype=np.float32)
        featVecs.append(concat)
    return featVecs

def hsvHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        hsv = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2HSV)
        h = np.histogram(hsv[:, :, 0], nBins, [0, 180])[0]
        s = np.histogram(hsv[:, :, 1], nBins, [0, 255])[0]
        v = np.histogram(hsv[:, :, 2], nBins, [0, 255])[0]
        concat = np.concatenate([h, s, v], axis=0)
        concat = concat / np.max(concat)
        concat = np.array(concat, dtype=np.float32)
        featVecs.append(concat)
    return featVecs

def labrgbHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        im = listIm[i]
        B = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
        g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
        r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l = np.histogram(lab[:, :, 0], nBins, [0, 255])[0]
        a = np.histogram(lab[:, :, 1], nBins, [0, 255])[0]
        b = np.histogram(lab[:, :, 2], nBins, [0, 255])[0]
        concat = np.concatenate([l,a, b,B,g,r], axis=0)
        concat = concat / np.max(concat)
        concat = np.array(concat, dtype=np.float32)
        featVecs.append(concat)
    return featVecs

def ycrcbHist(listIm,nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        im = listIm[i]
        ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        y = np.histogram(ycrcb[:, :, 0], nBins, [0, 180])[0]
        cr = np.histogram(ycrcb[:, :, 1], nBins, [0, 255])[0]
        cb = np.histogram(ycrcb[:, :, 2], nBins, [0, 255])[0]
        concat = np.concatenate([y, cr, cb], axis=0)
        concat = concat / np.max(concat)
        concat = np.array(concat, dtype=np.float32)
        featVecs.append(concat)
    return featVecs
