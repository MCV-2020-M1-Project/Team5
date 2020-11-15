import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calcBestMatches(matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    return good

def matchDescriptorsSIFT(desQ,desMList,imgQ,imgMList,kpQ,kpMList):
    allMatches = []
    plot = 0
    for i in range(len(desMList)):
        print('match:',i)
        matches = bf.knnMatch(desMList[i], desQ, k=2)
        bestM = calcBestMatches(matches)
        if plot:
            img3 = cv.drawMatchesKnn(imgMList[i], kpMList[i], imgQ, kpQ, bestM, None, flags=2)
            plt.imshow(img3), plt.show()
        if len(bestM)<=2:
            bestM = -1
        allMatches.append(bestM)
    return allMatches

def findMostMatchings(allMatches,idsMImages):
    numOfMatches = []
    for matches in allMatches:
        if matches == -1:
            numOfMatches.append(-1)
        else:
            numOfMatches.append(len(matches))
    idxS= np.argsort(numOfMatches)
    sortedMatchesId = np.array(numOfMatches)[idxS]
    sortedMatches = np.array(allMatches)[idxS]
    sortedIdIm = np.array(idsMImages)[idxS]
    sortedMatches = list(sortedMatches)
    sortedIdIm = list(sortedIdIm)
    #Find -1 in sortedMatches
    idx = [i for i, x in enumerate(sortedMatches) if x == -1]
    sortedIdIm = [i for j, i in enumerate(sortedIdIm) if j not in idx]
    sortedMatches = [i for j, i in enumerate(sortedMatches) if j not in idx]
    if sortedIdIm == []:
        sortedIdIm = [-1]
    else:
        sortedMatches.reverse()
        sortedIdIm.reverse()
    return sortedIdIm,sortedMatches
