import numpy as np
import math
import cv2



# Euclidean distance
def euclid_dist(vec1, vec2):
    d = np.linalg.norm(vec1-vec2)
    return d


#Chi square distance
def chiSquare_dist(vec1,vec2):
    d = cv2.compareHist(vec1, vec2, cv2.HISTCMP_CHISQR)
    return d


# Correlation
def corrDist(vec1, vec2):
    d = - cv2.compareHist(vec1, vec2, cv2.HISTCMP_CORREL)
    return d


# Histogram intersection (similarity)
def Hist_intersection(vec1, vec2):
    similarity = cv2.compareHist(vec1, vec2, cv2.HISTCMP_INTERSECT)
    d = -similarity
    return d


# Hellinger kernel (similarity)
def Hellinger_kernel(vec1, vec2):
    d = cv2.compareHist(vec1, vec2, cv2.HISTCMP_HELLINGER)
    return d