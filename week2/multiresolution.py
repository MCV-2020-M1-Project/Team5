from histograms import *
from utils import *
from calculateDistances import *
import pickle
import ml_metrics as metrics


def compute_multiresolution(image,listMImgs,idsMImgs,k):

    listMHists = []
    sumKdists = []

    # Divide image in k parts
    dividedImage = divideImg(image, k)

    # Calculate lab histograme for each of the subimages
    listQHists = labHist(dividedImage, nBins=64)    # lab space color

    # For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(listMImgs)):
        Kdists = []
        # Divide image in k parts
        dividedImage = divideImg(listMImgs[j], k)
        # Calculate histograme for each of the subimages
        listsubMHist = labHist(dividedImage, nBins=64)
        listMHists.append(listsubMHist)
        for m in range(k):
            dist = Hellinger_kernel(listQHists[m][:], listMHists[j][m][:]) # hellinger metric
            Kdists.append(dist)
        sumKdists.append(sum(Kdists))
    allDist, idsSorted = zip(*sorted(zip(sumKdists, idsMImgs)))

    return list(idsSorted)
