import cv2
from utils import *
from masks import *
from textDetection import *
import matplotlib.pyplot as plt
from multiresolution import *


# Params that should be modified
pathQ2 = 'qsd1_w2'  # Path to the query images
pathM2 = 'BBDD'
k = 10 # Number of bests predictions

listQImgs , idsQImgs = load_images(pathQ2)
listMImgs,idsMImgs = load_images(pathM2)

pkl_path = "result.pkl"
results = []
bbox = []

# For each cropped image, a textbox is seacrhed and removed and a multiresolution is computed
for img in listQImgs:

    img = [img] # convert list into a list of lists
    results_paiting = [] # init

    for painting in img: # loop for each individual painting from one image

        txtbox = calculate_txtbox(painting)
        x, y, w, h = txtbox
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

        results_paiting.append(klist)

    results.append(results_paiting)

## -----------------------------------------------------------------------------------------------
# Store the bounding boxes and find the MAP@K metric

filename = 'text_boxes.pkl'
outfile = open(filename, 'wb')
pickle.dump(bbox, outfile)
outfile.close()

gt_file = open('qsd1_w2/gt_corresps.pkl','rb') # it assumes that you are using qsd1_w2 in order to compute the MAP
gtquery_list = pickle.load(gt_file)
gt_file.close()

res = resultMAP(results)
mapkScore = metrics.mapk(gtquery_list, res, k)
