import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

def load_images(folder):
    images = []
    ids = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                ids.append(int(temp1))
    if not ids:
        ordImgs = None
        ordIds = None
    else:
        ordIds, ordImgs = zip(*sorted(zip(ids, images)))

    return ordImgs, ordIds


def divideImg(img, k):
    if k == 1:
        return img
    if k == 2:
        sR = np.size(img, 0)
        sR = int(np.floor(sR / (k)))
        sC = np.size(img, 1)
        sC = int(np.floor(sC / (k)))
        divIm = []
        for i in range(int(k)):
            for j in range(int(k)):
                div = img[i * sR:sR * (i + 1), j * sC:sC * (j + 1), :]
                divIm.append(div)
        return divIm

    else:
        if int(math.sqrt(k)) % k == 0:
            nCol = int(math.sqrt(k))
            nRow = int(math.sqrt(k))
        else:
            nCol = int(math.sqrt(k))
            nRow = k // nCol

        if nCol * nRow == k:
            sR = np.size(img, 0)
            sR = int(np.floor(sR // (nRow)))
            sC = np.size(img, 1)
            sC = int(np.floor(sC // (nCol)))
            divIm = []
            for i in range(int(nRow)):
                for j in range(int(nCol)):
                    div = img[i * sR:sR * (i + 1), j * sC:sC * (j + 1), :]
                    divIm.append(div)
        else:
            print('Warning: K must not be a prime number, try it with another number. Good luck! :)')
            divIm = None
    return divIm


def filterContours(cont,im,imbin):
    contNew = []
    difX = []
    difY = []
    minXa = []
    minYa = []
    x = 0
    y = 0
    w = 1
    h = 1
    if cont == []:
        return 0,0,1,1
    f = bestAr(im)
    A = (np.size(im, 0) * np.size(im, 1))
    for iC in range(0, len(cont)):
        if cont[iC].shape[0] > 3:
            minX = min(cont[iC][:,0,0])
            minXa.append(minX)
            maxX = max(cont[iC][:,0,0])
            minY = min(cont[iC][:,0,1])
            minYa.append(minY)
            maxY = max(cont[iC][:,0,1])
            difX.append(maxX-minX)
            difY.append(maxY-minY)
            contNew.append(cont[iC])
    if difY == []:
        return 0,0,1,1
    difT = np.array(difX)*np.array(difY)
    indMax = np.argmax(difT)
    for i in range(len(difT)):
        if np.max(difT)<=(1/f)*A:
            indMax = np.argmax(difT)
            x = minXa[indMax]
            y = minYa[indMax]
            w = difX[indMax]
            h = difY[indMax]

            if (difX[indMax]/difY[indMax])>1:
                break
            else:
                ind = np.argmax(difT)
                difT[ind] = 0
        else:
            ind = np.argmax(difT)
            difT[ind] = 0


    return x, y, w, h

def bestAr(im):
    if (np.size(im,0)+150)<np.size(im,1):
        f = 2
    else:
        f = 4
    return f

def chooseBestBbox(w,h):
    ar = []
    for i in range(len(w)):
        ar.append(w[i]/h[i])
    ind = np.argmax(ar)
    if ar[ind]>8:
        ar[ind] = 0
        ind = np.argmax(ar)
    return ind

def histocount(histr):
    cont = 0
    for i in range(len(histr)):
        if histr[i]!= 0:
            cont +=1
    return cont

def histInBbox(x,y,w,h,im):
    roi = im[y:y+h,x:x+w]
    kernel = np.ones((7, 7), np.float32) / 25
    roi = cv2.filter2D(roi, -1, kernel)
    histr = np.histogram(roi,bins=np.arange(256))
    count = histocount(histr[0])
    return count

def iou(boxA,boxB):
    #Coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    #Compute area intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #Compute area of input bboxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #Calculate iou: areaIntersection/area of union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def convertBbox(gtBox):
    gt = []
    for i in range(len(gtBox)):
        a = gtBox[i][0][0][0]
        b = gtBox[i][0][2][0]
        x = np.min([a,b])
        x_ = np.max([a,b])
        a = gtBox[i][0][0][1]
        b = gtBox[i][0][2][1]
        y = np.min([a, b])
        y_ = np.max([a, b])
        gt.append([x,y,x_,y_])
    return gt



def findArea(contours,im):
    bigShapes = []
    A = (np.size(im,0)*np.size(im,1))
    for i in range(len(contours)):
        if np.shape(contours[i])[0] == 4:
            print('hi ha quadrilater')
            base = abs(contours[i][2][0][0]-contours[i][0][0][0])
            altura = abs(contours[i][2][0][1]-contours[i][0][0][1])
            area = base*altura
            if area>=(1/60)*A:
                bigShapes.append(contours)
    return bigShapes

def load_masks(folder):
    images = []
    ids = []
    for filename in os.listdir(folder):
        if filename.split('.')[1] == 'png':
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                images.append(img)
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                ids.append(int(temp1))
    if not ids:
        ordMasks = None
        ordIds = None
    else:
        ordIds, ordMasks = zip(*sorted(zip(ids, images)))

    return ordMasks, ordIds