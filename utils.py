import cv2
import os
import numpy as np

# Generic functions
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
                for i in range(len(temp1)):
                    id = None
                    if temp1[i] != '0':
                        id = temp1[i:]
                        break
                if id is None:
                    id = '0'
                ids.append(id)
    idx = np.argsort(ids)
    ordIds = np.array(ids)[idx]
    ordImages = np.array(images)[idx]

    return ordImages, ordIds


def sort_ids(metric,idList):
    id = np.argsort(metric)
    sortMetric = np.array(metric)[id]
    sortedIds = np.array(idList)[id]
    return sortMetric, list(map(int, sortedIds))


def shift_hist_(hist,og,left):
    difference = abs(list(hist).index(max(hist))-og)
    z = np.zeros(difference-1)

    if left:
        newHist_ = hist[difference:]
        newHist = np.concatenate(newHist_,z)
    else:
        newHist_ = hist[:difference]
        newHist = np.concatenate((z,newHist_))

    return newHist


def center_hist(hist1,hist2):

    if np.size(hist1[:][:][0]) == 1:
        difference = list(hist2).index(max(hist2)) - list(hist1).index(max(hist1))
        if difference > 2 or difference < -2:
            z = np.zeros(abs(difference) - 1)
            if difference<0:
                hist2 = np.roll(hist2, difference)
                hist2[(len(hist2) - abs(difference) + 1):] = z
            else:
                hist2 = np.roll(hist2, difference)
                hist2[:difference - 1] = z
            hist1 = hist1

        else:
            hist1 = hist1
            hist2 = hist2


    return hist1,hist2

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






