import cv2
import numpy as np
import os


def calculate_gray_hist(listIm):
    histGray = []
    for img in listIm:
        print(img.max())
        print(img.shape)
        # cv2.imshow('og', img)
        # cv2.waitKey()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', imgGray)
        # cv2.waitKey()
        histr = cv2.calcHist([imgGray], [0], None, [256], [0, 256])
        histr = cv2.normalize(histr,histr)
        histGray.append(histr)
        # plt.plot(histr)
        # plt.xlim([0, 256])
    return histGray


def labHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l = np.histogram(lab[:, :, 0], nBins, [0, 255])[0]
            a = np.histogram(lab[:, :, 1], nBins, [0, 255])[0]
            b = np.histogram(lab[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([l,a, b], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids, featVecs)))
    return featVecs, ids

def labrgbHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            B = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
            g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
            r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l = np.histogram(lab[:, :, 0], nBins, [0, 255])[0]
            a = np.histogram(lab[:, :, 1], nBins, [0, 255])[0]
            b = np.histogram(lab[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([l,a, b,B,g,r], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids, featVecs)))
    return featVecs, ids


def hsvrgbHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            B = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
            g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
            r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
            im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h = np.histogram(im_hsv[:, :, 0], nBins, [0, 180])[0]
            s = np.histogram(im_hsv[:, :, 1], nBins, [0, 255])[0]
            v = np.histogram(im_hsv[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([h,s, v,B,g,r], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids, featVecs)))
    return featVecs, ids


def rgbHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            b = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
            g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
            r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([b, g, r], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids, featVecs)))
    return featVecs, ids


def ycrcbHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
            y = np.histogram(ycrcb[:, :, 0], nBins, [0, 180])[0]
            cr = np.histogram(ycrcb[:, :, 1], nBins, [0, 255])[0]
            cb = np.histogram(ycrcb[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([y, cr, cb], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids, featVecs)))
    return featVecs, ids


def hsvHist(path,nBins):
    featVecs = []
    ids = []
    iIm = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            iIm += 1
            print(str(iIm))
            im = cv2.imread(os.path.join(path, file))
            im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h = np.histogram(im_hsv[:, :, 0], nBins, [0, 180])[0]
            s = np.histogram(im_hsv[:, :, 1], nBins, [0, 255])[0]
            v = np.histogram(im_hsv[:, :, 2], nBins, [0, 255])[0]
            featVecs.append(np.concatenate([h, s, v], axis=0))
            if '_' in file:
                file = file.split('_')[1]
            name = file.split('.')[0]
            ids.append(int(name))
    ids, featVecs = zip(*sorted(zip(ids,featVecs)))
    return featVecs,ids

#-----------------------------------------Hists for task 2 ( images stored in list)-------------------------------------

def labHist2(listIm,nBins,id):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        print(str(iIm))
        hsv = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2LAB)
        l = np.histogram(hsv[:, :, 0], nBins, [0, 255])[0]
        a = np.histogram(hsv[:, :, 1], nBins, [0, 255])[0]
        b = np.histogram(hsv[:, :, 2], nBins, [0, 255])[0]
        featVecs.append(np.concatenate([l, a, b], axis=0))
    ids, featVecs = zip(*sorted(zip(id, featVecs)))
    return featVecs, ids

def hsvHist2(listIm,nBins,id):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        print(str(iIm))
        hsv = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2HSV)
        h = np.histogram(hsv[:, :, 0], nBins, [0, 180])[0]
        s = np.histogram(hsv[:, :, 1], nBins, [0, 255])[0]
        v = np.histogram(hsv[:, :, 2], nBins, [0, 255])[0]
        featVecs.append(np.concatenate([h, s, v], axis=0))
    ids, featVecs = zip(*sorted(zip(id, featVecs)))
    return featVecs, ids

def labrgbHist2(listIm,nBins,id):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        print(str(iIm))
        im = listIm[i]
        B = np.histogram(im[:, :, 0], nBins, [0, 255])[0]
        g = np.histogram(im[:, :, 1], nBins, [0, 255])[0]
        r = np.histogram(im[:, :, 2], nBins, [0, 255])[0]
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l = np.histogram(lab[:, :, 0], nBins, [0, 255])[0]
        a = np.histogram(lab[:, :, 1], nBins, [0, 255])[0]
        b = np.histogram(lab[:, :, 2], nBins, [0, 255])[0]
        featVecs.append(np.concatenate([l,a, b,B,g,r], axis=0))
    ids, featVecs = zip(*sorted(zip(id, featVecs)))
    return featVecs, ids