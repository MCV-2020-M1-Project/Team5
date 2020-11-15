import numpy as np
import os
import cv2
import math
from math import log10, sqrt

# input: folder path // output: list of images sorted by ID, list of IDs
def load_images(folder):
    images = []
    ids = []
    for filename in os.listdir(folder):
        if "." not in filename: continue
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

def load_authors(folder):
    authors = []
    ids = []
    for filename in os.listdir(folder):
        if "." not in filename: continue
        if filename.split('.')[1] == 'txt':
            path = os.path.join(folder, filename)
            f = open(path, 'r')
            txt_str = f.read()
            txt_str = txt_str.replace('\n', '')
            f.close()

            if txt_str != '':
                name = txt_str.split("'")
                name = name[1]
                authors.append(name)
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                ids.append(int(temp1))
            else:
                authors.append('Unknown')
                if '_' in filename:
                    temp = filename.split('_')[1]
                else:
                    temp = filename
                temp1 = temp.split('.')[0]
                ids.append(int(temp1))

    if not ids:
        ordAuthors = None
        ordIds = None
    else:
        ordIds, ordAuthors = zip(*sorted(zip(ids, authors)))

    return ordAuthors, ordIds

# input: imgage, number of divisions (ex: 2 = divided in 2x2 regions) // output: list of divided images
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

# It transforms the set of point of the contours into bounding boxes
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
    bestX = []
    bestY = []
    ar  = []
    indexbest = []
    surface = []
    for i in range(len(difT)):
        if np.max(difT)<=(1/f)*A:
            indMax = np.argmax(difT)
            x = minXa[indMax]
            y = minYa[indMax]
            w = difX[indMax]
            h = difY[indMax]

            if ((difX[indMax]/difY[indMax])>2) and ((difX[indMax]/difY[indMax])<15):
                bestX.append(difX[indMax])
                bestY.append(difY[indMax])
                ar.append(difX[indMax]/difY[indMax])
                surface.append(difX[indMax]*difY[indMax])
                indexbest.append(indMax)
                difT[indMax] = 0

            else:
                ind = np.argmax(difT)
                difT[indMax] = 0
        else:
            ind = np.argmax(difT)
            difT[ind] = 0
    for j in range(len(bestX)):
        a = np.argmax(surface)
        if (bestY[a]<=np.size(im,0)*0.5):
            ind = indexbest[a]
            x = minXa[ind]
            y = minYa[ind]
            w = difX[ind]
            h = difY[ind]
            break
        else:
            surface[a] = 0

    return x, y, w, h

# Choose the best Aspect Ratio
def bestAr(im):
    if (np.size(im,0)+150)<np.size(im,1):
        f = 2
    else:
        f = 4
    return f

# Choose the most suitable bounding box (rectangular, not too big or too small compared with the original image)
def chooseBestBbox(w,h):
    ar = []
    for i in range(len(w)):
        ar.append(w[i]/h[i])
    ind = np.argmax(ar)
    if ar[ind]>12:
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

# Computes the intersection over union metric
def iou(boxA,boxB):
    if np.size(boxB[0]) == 4:
        boxB = boxB[0]
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

# Solving the errors of the ground truth bounding box
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

# load the masks of the DDBB to do the evaluation
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


# it gets the result pkl and returns the map value
def resultMAP(result):
    r = []
    for i in range(np.size(result,0)):
        r.append(result[i][0])
        if len(result[i]) == 2:
            r.append(result[i][1])
    return r

# it corrects the way it is stored the GT (each image as a list of lists, as it is said in the slides)
def transformGT(gt_list):
    gt = []
    for i in range(len(gt_list)):
        gt.append([gt_list[i][0]])
        if len(gt_list[i]) == 2:
            gt.append([gt_list[i][1]])
    return gt

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr