from textDetection import *
from descriptors import *
from masks import *
import numpy as np
import cv2 as cv


# Params that should be modified
pathQ3 = 'qst2_w3'  # Path to the query images
pathM = 'BBDD'      # Path to the Database
k = 10              # Number of bests predictions

# Load the list of images
listQImgs, idsQImgs = load_images(pathQ3)
listMImgs, idsMImgs = load_images(pathM)

# Load the list of the authors of the paintings sorted with idsMImgs
listMAuthors, idsMAuthors = load_authors(pathM)

# Load the GT of the text boundary boxes (not needed as a deliverable)
if os.path.exists(os.path.join(pathQ3,'text_boxes.pkl')):
    with open(os.path.join(pathQ3,'text_boxes.pkl'), 'rb') as f:
        gtBbox = pickle.load(f)
else:
    gtBbox = None
allIou = []

# Init lists of the results
results = []
bbox = []

# Crop images for each painting
cropped_images = crop_imgarray(listQImgs)

# Create a folder to store the name of the authors
dir = 'authors'
if not os.path.exists(dir):
    os.mkdir(dir)


# --- Loop for all images of the Query ---
i = 0 # counter for debugging purposes
for img in cropped_images:
    results_painting = []
    box_painting = []
    print(i) # counter for debugging purposes

    f = open('authors/000' + str(i) + '.txt', "a+")

    for painting in img:  # loop for each individual painting from one image

        # Compute the denoising filter for each image channel
        denoised_img = np.zeros((np.size(painting, 0), np.size(painting, 1), np.size(painting, 2)))
        denoised_img[:, :, 0] = cv.medianBlur(painting[:, :, 0], 3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 1] = cv.medianBlur(painting[:, :, 1], 3)  # Median is the best effective for salt and pepper noise
        denoised_img[:, :, 2] = cv.medianBlur(painting[:, :, 2], 3)  # Median is the best effective for salt and pepper noise
        denoised_img = denoised_img.astype(np.uint8)

        # Calculate the bbox of the text
        gtBox_i = None
        if gtBbox is not None:
            gtBox_i = gtBbox[i]
        iou_, [x, y, w, h] = calculate_txtbox(denoised_img, gtBox_i)
        if iou_ is not None:
            allIou.append(iou_)
        Bbox = [x, y, x+w, y+h]
        box_painting.append(Bbox) # Add the results list

        # Text descriptor, returns the ids sorted by the distance
        idsSorted, max_paintings, name = text_descriptor(denoised_img, Bbox, listMAuthors, idsMImgs)
        f.write("%s\n" % name)  # write the name to a txt

        # Ignore the text box
        mask = np.ones(denoised_img.shape[:2], dtype="uint8")
        mask[y:y + h, x:x + w] = 0

        # Color descriptor only to the top images, returns the ids sorted by the distance
        best_idsMImgs = idsSorted[:max_paintings]
        best_listMImgs = []
        for j in range(0, max_paintings):
            id = idsSorted[j]
            best_listMImgs.append(listMImgs[id])
        best_idsSorted = color_descriptor(denoised_img, mask, best_listMImgs, best_idsMImgs, 4)
        # Complete the list of k bests
        if max_paintings < k:
            best_idsSorted = best_idsSorted + idsSorted[max_paintings:k]
        else:
            best_idsSorted[:k]

        # Color descriptor, returns the ids sorted by the distance
        #idsSorted = color_descriptor(denoised_img, mask, listMImgs, idsMImgs, 4)

        # Texture descriptor, returns the ids sorted by the distance
        #idsSorted = texture_descriptor(denoised_img, listMImgs, idsMImgs, 4, 'LBPSci')

        # Only store the k bests
        kList = list(best_idsSorted)
        kList = kList[:k]

        results_painting.append(kList)  # painting list

    i = i + 1
    bbox.append(box_painting)           # final bbox list
    results.append(results_painting)    # final img list
    f.close()

## -----------------------------------------------------------------------------------------------
# Store the bounding boxes, results and find the MAP@K metric

# pkl_path = "result.pkl"
# outfile = open(pkl_path, 'wb')
# pickle.dump(results, outfile)
# outfile.close()
#
# pkl_path = 'text_boxes.pkl'
# outfile = open(pkl_path, 'wb')
# pickle.dump(bbox, outfile)
# outfile.close()
#
# gt_file = open('qsd2_w3/gt_corresps.pkl','rb') # it assumes that you are using qsd1_w3 in order to compute the MAP
# gtquery_list = pickle.load(gt_file)
# gt_file.close()
# gtquery_list_corrected = transformGT(gtquery_list)
#
# # Correct predictions array to compute correctly the MAPKScore
# for i in range(0,len(results)):
#     aux1 = len(gtquery_list[i])
#     aux2 = len(results[i])
#     if aux1 < aux2:
#         for j in range(aux2 - aux1):
#             results[i].pop(-1)
#     elif aux2 > aux1:
#         for j in range(aux1 - aux2):
#             results[i].append(np.zeros[k])
#
# res = resultMAP(results)
# mapkScore = metrics.mapk(gtquery_list_corrected, res, 5)
# print('MAP@5: ' + str(mapkScore))
#
# mapkScore = metrics.mapk(gtquery_list_corrected, res, 1)
# print('MAP@1: ' + str(mapkScore))