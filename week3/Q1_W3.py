from textDetection import *
from descriptors import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# Params that should be modified
pathQ3 = 'qsd1_w3'  # Path to the query images
pathM = 'BBDD'      # Path to the Database
k = 10 # Number of bests predictions

# Load the list of images
listQImgs, idsQImgs = load_images(pathQ3)
listMImgs, idsMImgs = load_images(pathM)

# Load the list of the authors of the paintings sorted with idsMImgs
listMAuthors, idsMAuthors = load_authors(pathM)

# Load the GT of the text boundary boxes (not needed as a deliverable)
if os.path.exists(os.path.join('qsd1_w3','text_boxes.pkl')):
    with open(os.path.join('qsd1_w3','text_boxes.pkl'), 'rb') as f:
        gtBbox = pickle.load(f)
else:
    gtBbox = None
allIou = []

# Init lists of the results
results = []
bbox = []


# Loop for all images of the Query
for i in range(0,len(listQImgs)):
    img = listQImgs[i]
    img = [img]  # convert list into a list of lists
    results_painting = []
    box_painting = []
    print(i)

    for painting in img:  # loop for each individual painting from one image

        # Compute the denoising filter for each image channel
        denoised_img = np.zeros((np.size(painting, 0), np.size(painting, 1), np.size(painting, 2)))
        denoised_img[:, :, 0] = cv.medianBlur(painting[:, :, 0], 3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 1] = cv.medianBlur(painting[:, :, 1], 3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 2] = cv.medianBlur(painting[:, :, 2], 3)  # Median is the best effective for salt and pepper noise
        denoised_img = denoised_img.astype(np.uint8)

        # Calculate the bbox of the text
        iou_, [x, y, w, h] = calculate_txtbox(denoised_img, gtBbox[i])
        if iou_ is not None:
            allIou.append(iou_)
        Bbox = [x, y, x+w, y+h]
        box_painting.append(Bbox) # Add the results list

        # Text descriptor, returns the ids sorted by the distance
        #idsSorted = text_descriptor(denoised_img, Bbox, listMAuthors, idsMImgs)

        # Ignore the text box
        mask = np.ones(denoised_img.shape[:2], dtype="uint8")
        mask[y:y + h, x:x + w] = 0

        # Color descriptor, returns the ids sorted by the distance
        idsSorted = color_descriptor(denoised_img, mask, listMImgs, idsMImgs, 4)

        # Texture descriptor, returns the ids sorted by the distance
        #idsSorted = texture_descriptor(denoised_img, listMImgs, idsMImgs, 4, 'LBPSci')

        # Only store the k bests
        kList = list(idsSorted)
        kList = kList[:k]

        results_painting.append(kList)  # painting list

    bbox.append(box_painting)           # final bbox list
    results.append(results_painting)    # final img list

## -----------------------------------------------------------------------------------------------
# Store the bounding boxes and find the MAP@K metric

#pkl_path = "result.pkl"
#outfile = open(pkl_path, 'wb')
#pickle.dump(results, outfile)
#outfile.close()

#pkl_path = 'text_boxes.pkl'
#outfile = open(pkl_path, 'wb')
#pickle.dump(bbox, outfile)
#outfile.close()

gt_file = open('qsd1_w3/gt_corresps.pkl','rb') # it assumes that you are using qsd1_w3 in order to compute the MAP
gtquery_list = pickle.load(gt_file)
gt_file.close()

res = resultMAP(results)
mapkScore = metrics.mapk(gtquery_list, res, 1)
print('MAP: ' + str(mapkScore))