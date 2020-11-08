import cv2 as cv
from textReader import textReader
from utils import *

# Text Similarity
def text_descriptor(image, Bbox, listMAuthors, idsMImgs):
    names = []
    # Read as a string the text from the image
    denoised_imgGray = cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2GRAY)
    textReader_obj = textReader(denoised_imgGray, Bbox)

    # Compute the similarities between the strings and select the name that gives the best ratio
    ratios = []
    name_readed1 = textReader_obj.cropped_text
    name_readed2 = textReader_obj.opening_text
    name_readed3 = textReader_obj.closing_text
    for j in range(0, len(listMAuthors)):
        name_DDBB = listMAuthors[j]

        if name_readed1 != '':
            ratio1 = levenshtein_ratio_and_distance(name_DDBB, name_readed1, ratio_calc=True)
        else:  # error case
            ratio1 = 0

        if name_readed2 != '':
            ratio2 = levenshtein_ratio_and_distance(name_DDBB, name_readed2, ratio_calc=True)
        else:  # error case
            ratio2 = 0

        if name_readed3 != '':
            ratio3 = levenshtein_ratio_and_distance(name_DDBB, name_readed3, ratio_calc=True)
        else:  # error case
            ratio3 = 0

        ratio = max(ratio1, ratio2, ratio3)

        # return the name
        if ratio1 >= ratio2 and ratio1 >= ratio3:
            names.append(name_readed1)
        elif ratio2 >= ratio1 and ratio2 >= ratio3:
            names.append(name_readed2)
        elif ratio3 >= ratio2 and ratio3 >= ratio1:
            names.append(name_readed3)

        ratios.append(ratio)

    # Obtain the ids of the paintings of the author that achieved the best ratio
    m = max(ratios)
    painting_ids = [j for j, k in enumerate(ratios) if k == m]
    max_paintings = len(painting_ids) # since they are sorted in idsSorted, we can just take the number of maximums

    # Create the sorted list
    ratiosSorted, idsSorted = zip(*sorted(zip(ratios, idsMImgs), reverse=True))

    # return the name
    final_name = names[idsSorted[0]]

    return idsSorted, max_paintings, final_name


def orb_descriptor(Qimage, mask, Mkp, Mdes,listMImgs,idsMImgs):

    kp_threshold = 20

    orb = cv2.ORB_create()
    Qkp, Qdes = orb.detectAndCompute(Qimage, mask)

    # Set method for descriptor matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # bf = cv.BFMatcher()

    allMatches = []
    bestMsum = []
    # For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(Mkp)):

        # For each museum image, get keypoint and compute descriptors
        if not Mkp[j] or not Qkp:
            bestMsum.append(0)
            bestM = []
        else:
            # matches = bf.knnMatch(Qdes,Mdes[j], k=2)
            # bestM = calcBestMatches(matches)
            matches = bf.match(Qdes,Mdes[j])
            bestM = [i for i in matches if i.distance < 40]


            bestMsum.append(len(bestM))
        allMatches.append(bestM)

    if max(bestMsum) >= kp_threshold:
        allMatchNum, idsSorted = zip(*sorted(zip(bestMsum, idsMImgs),reverse=True))
    else:
        idsSorted = tuple([-1])

    return idsSorted


def sift_descriptor(Qimage, mask, Mkp, Mdes,listMImgs,idsMImgs):

    kp_threshold = 20

    sift = cv2.SIFT_create(nfeatures=100)
    Qkp, Qdes = sift.detectAndCompute(Qimage, mask)

    # Set method for descriptor matching
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    bf = cv.BFMatcher()

    allMatches = []
    bestMsum = []
    # For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(Mkp)):

        # For each museum image, get keypoint and compute descriptors
        if not Mkp[j] or not Qkp:
            bestMsum.append(0)
            bestM = []
        else:
            matches = bf.knnMatch(Qdes,Mdes[j], k=2)
            bestM = calcBestMatches(matches)
            # matches = bf.match(Qdes,Mdes[j])
            # bestM = [i for i in matches if i.distance < 40]

            bestMsum.append(len(bestM))
        allMatches.append(bestM)

    if max(bestMsum) >= kp_threshold:
        allMatchNum, idsSorted = zip(*sorted(zip(bestMsum, idsMImgs),reverse=True))
    else:
        idsSorted = tuple([-1])

    return idsSorted


def calcBestMatches(matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    return good