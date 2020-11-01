import cv2 as cv
from textReader import textReader
from utils import *
from histograms import *
from calculateDistances import *


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

# Multiresolution
def color_descriptor(image, mask, listMImgs, idsMImgs, divisions):
    listMHists = []
    sumKdists = []

    # Divide image in k parts
    dividedImageQ = divideImg(image, divisions)
    dividedImageQ = np.float32(dividedImageQ)
    mask_aux = image
    mask_aux[:, :, 0] = mask
    mask_aux[:, :, 1] = mask
    mask_aux[:, :, 2] = mask
    dividedMask = divideImg(mask_aux, divisions)
    dividedMask = np.uint8(dividedMask)
    dividedMask = dividedMask[:,:,:,0]

    # Calculate lab histogram for each of the subimages
    listsubQHists = labHistMask(dividedImageQ.astype(np.uint8), dividedMask, nBins=64)  # lab space color

    # For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(listMImgs)):
        Kdists = []
        # Divide image in k parts
        DDBBimage = listMImgs[j].astype(np.uint8)
        dividedImageM = divideImg(DDBBimage, divisions)
        # Calculate histogram for each of the subimages
        listsubMHist = labHist(dividedImageM, nBins=64)
        listMHists.append(listsubMHist)
        for m in range(divisions):
            dist = Hellinger_kernel(listsubQHists[m][:], listsubMHist[m][:])    # hellinger metric
            Kdists.append(dist)
        sumKdists.append(sum(Kdists))
    allDist, idsSorted = zip(*sorted(zip(sumKdists, idsMImgs)))

    return idsSorted

# LBP
def texture_descriptor(image, listMImgs, idsMImgs, divisions, type='LBP'):  # type -> ['LBP', 'LBPSci', 'HOG']
    listMHists = []
    sumKdists = []

    # Divide image in k parts
    dividedImageQ = divideImg(image, divisions)
    dividedImageQ = np.float32(dividedImageQ)

    listsubQHists = []
    if type == 'LBP':
        listsubQHists = lbpHist(dividedImageQ.astype(np.uint8), nBins=64)
    elif type == 'LBPSci':
        listsubQHists = lbpHistSci(dividedImageQ.astype(np.uint8), nBins=64)
    elif type == 'HOG':
        listsubQHists = 0

    # For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(listMImgs)):
        Kdists = []
        # Divide image in k parts
        DDBBimage = listMImgs[j].astype(np.uint8)
        dividedImageM = divideImg(DDBBimage, divisions)
        # Calculate histograme for each of the subimages
        listsubMHist = []
        if type == 'LBP':
            listsubMHist = lbpHist(dividedImageM, nBins=64)
        elif type == 'LBPSci':
            listsubMHist = lbpHistSci(dividedImageM, nBins=64)
        elif type == 'HOG':
            listsubMHist = 0

        listMHists.append(listsubMHist)
        for m in range(divisions):
            dist = Hellinger_kernel(listsubQHists[m][:], listsubMHist[m][:])    # hellinger metric
            Kdists.append(dist)
        sumKdists.append(sum(Kdists))
    allDist, idsSorted = zip(*sorted(zip(sumKdists, idsMImgs)))

    return idsSorted