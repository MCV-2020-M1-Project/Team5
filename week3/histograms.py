from feature_extraction import *


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

def labHistMask(listIm, mask, nBins):
    featVecs = []
    iIm = 0
    for i in range(len(listIm)):
        iIm += 1
        mask_aux = mask[i]
        lab = cv2.cvtColor(listIm[i], cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.uint8)
        l = cv2.calcHist([lab], [0], mask_aux, [nBins], [0, 255])
        a = cv2.calcHist([lab], [1], mask_aux, [nBins], [0, 255])
        b = cv2.calcHist([lab], [2], mask_aux, [nBins], [0, 255])
        concat = np.concatenate([l, a, b], axis=0)
        concat = concat / np.max(concat)
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


def lbpHist(listIm, nBins):
    featVecs = []
    for i in range(len(listIm)):
        img = listIm[i]
        img_lbp = LBP(img)
        lbp = np.histogram(img_lbp, nBins, [0, 255])[0]
        lbp = lbp / np.max(lbp)
        lbp = np.array(lbp, dtype=np.float32)
        featVecs.append(lbp)

    return featVecs


def lbpHistSci(listIm,nBins):

    featVecs = []
    radius = 2
    numPoints = 8
    eps = 1e-7

    for i in range(len(listIm)):
        im = listIm[i]
        lbp = LBP_sci(im,radius,numPoints)

        (hist, _) = np.histogram(lbp.ravel(),bins=numPoints+2,range=(0, numPoints + 2))
        hist = hist / np.max(hist)
        hist = np.array(hist, dtype=np.float32)

        featVecs.append(hist)

    return featVecs


def hogHist(listIm, nBins):

    hog = cv2.HOGDescriptor()
    featVecs = []

    for i in range(len(listIm)):
        im = listIm[i]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h = hog.compute(im)
        hog_hist = np.histogram(h, nBins, [0, 1])[0]
        hog_hist = np.array(hog_hist, dtype=np.float32)
        featVecs.append(hog_hist)

    return featVecs
