import cv2
from utils import *
from masks import *
from textDetection import *
import matplotlib.pyplot as plt
from multiresolution import *


# Params that should be modified
pathQ2 = 'qst2_w2'  # Path to the query images
pathM2 = 'BBDD'
k = 10 # Number of bests predictions

listQImgs , idsQImgs = load_images(pathQ2)
listMImgs,idsMImgs = load_images(pathM2)

pkl_path = "result.pkl"
results = []
bbox = []

# Crop images for each painting
crped_images = crop_imgarray(listQImgs)

# For each cropped image, a textbox is seacrhed and removed and a multiresolution is computed
for img in crped_images:

    results_painting = []
    box_painting = []

    for painting in img: # loop for each individual painting from one image

        [x, y, w, h] = calculate_txtbox(painting)
        box_painting.append([x, y, x+w, y+h])

        mask = np.ones(painting.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = -1

        d3mask = np.zeros_like(painting)
        d3mask[:,:,0] = mask
        d3mask[:,:,1] = mask
        d3mask[:,:,2] = mask

        painting_tbx = painting * d3mask

        # Returns ids of Museum images sorted by similarity
        klist = compute_multiresolution(painting_tbx,listMImgs,idsMImgs,4)
        klist = klist[:k]

        results_painting.append(klist)

    bbox.append(box_painting)
    results.append(results_painting)

## -----------------------------------------------------------------------------------------------
# Store the bounding boxes and find the MAP@K metric

outfile = open(pkl_path, 'wb')
pickle.dump(results, outfile)
outfile.close()

filename = 'text_boxes.pkl'
outfile = open(filename, 'wb')
pickle.dump(bbox, outfile)
outfile.close()

gt_file = open('qsd2_w2/gt_corresps.pkl','rb') # it assumes that you are using qsd2_w2 in order to compute the MAP
gtquery_list = pickle.load(gt_file)
gt_file.close()
gtquery_list_corrected = transformGT(gtquery_list)

res = resultMAP(results)
mapkScore = metrics.mapk(gtquery_list_corrected, res, 5)
print(mapkScore)

mapkScore = metrics.mapk(gtquery_list_corrected, res, 1)
print(mapkScore)