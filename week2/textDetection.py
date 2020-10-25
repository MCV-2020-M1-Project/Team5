from histograms import *
from utils import *
from calculateDistances import *
import cv2
import pickle
import ml_metrics as metrics


def calculate_txtbox(image):

    # Params to change
    kernel = np.ones((7, 7), np.uint8)
    kernel1 = np.ones((4, 4), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)

    #Transform image to grayscale:
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    opening = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel)
    gradient = (opening-closing)/np.max(opening-closing) # normalize to apply the thr
    #LPF to get rid of high freq
    gradient = cv2.filter2D(gradient, -1, kernel)
    #Binarize the gradient
    retval, binaryGrad = cv2.threshold(gradient.astype(np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Filter again to get rid of the noise of the binarization
    binaryGradF = cv2.filter2D(binaryGrad, -1, kernel)
    #Apply morphological operations
    binaryGrad1 = cv2.morphologyEx(binaryGradF, cv2.MORPH_OPEN, kernel1)
    binaryGrad2 = cv2.morphologyEx(binaryGradF, cv2.MORPH_CLOSE, kernel1)
    binaryGrad3 = cv2.morphologyEx(binaryGrad1, cv2.MORPH_CLOSE, kernel1)
    binaryGrad4 = cv2.morphologyEx(binaryGrad2, cv2.MORPH_OPEN, kernel1)
    binaryGrad5 = cv2.morphologyEx(binaryGradF, cv2.MORPH_OPEN, kernel2)
    binaryGrad6 = cv2.morphologyEx(binaryGradF, cv2.MORPH_CLOSE, kernel2)
    binaryGrad5 = cv2.erode(binaryGrad5, kernel, iterations=2)
    binaryGrad6 = cv2.dilate(binaryGrad6, kernel, iterations=2)

    im1 = binaryGrad1.astype(np.uint8)
    im2 = binaryGrad2.astype(np.uint8)
    im3 = binaryGrad3.astype(np.uint8)
    im4 = binaryGrad4.astype(np.uint8)
    im5 = binaryGrad5.astype(np.uint8)
    im6 = binaryGrad6.astype(np.uint8)
    #Compute point contours
    contours1, thr = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, thr = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, thr = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours4, thr = cv2.findContours(im4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours5, thr = cv2.findContours(im5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours6, thr = cv2.findContours(im6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Create bboxes
    x1, y1, w1, h1 = filterContours(contours1,imgGray,im1)
    x2, y2, w2, h2 = filterContours(contours2, imgGray,im2)
    x3, y3, w3, h3 = filterContours(contours3, imgGray,im3)
    x4, y4, w4, h4 = filterContours(contours4, imgGray,im4)
    x5, y5, w5, h5 = filterContours(contours5, imgGray,im5)
    x6, y6, w6, h6 = filterContours(contours6, imgGray,im6)

    W = [w1,w2,w3,w4,w5,w6]
    H = [h1,h2,h3,h4,h5,h6]
    X = [x1,x2,x3,x4,x5,x6]
    Y = [y1,y2,y3,y4,y5,y6]
    ind = chooseBestBbox(W,H)
    return [X[ind],Y[ind],X[ind]+W[ind],Y[ind]+H[ind]]

#
#
# #Params to change
# qPath = 'qsd1_w2'
# mPath = 'BBDD'
# kernel = np.ones((7,7),np.uint8)
# kernel1 = np.ones((4,4),np.uint8)
# kernel2 = np.ones((10,10),np.uint8)
# plots = 0
#
# listQImgs , idsQImgs = load_images(qPath)
# #listMImgs , idsMImgs = load_images(mPath)
# if os.path.exists(os.path.join(qPath,'text_boxes.pkl')):
#     with open(os.path.join(qPath,'text_boxes.pkl'), 'rb') as f:
#         gtBbox = pickle.load(f)
#     # Fix bbox format
#     gtBbox = convertBbox(gtBbox)
# allIou = []
# for i in range(0,len(listQImgs)):
#
#     #Transform image to grayscale:
#     imgGray = cv2.cvtColor(listQImgs[i], cv2.COLOR_BGR2GRAY)
#     opening = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel)
#     gradient = (opening-closing)/np.max(opening-closing) # normalize to apply the thr
#     #LPF to get rid of high freq
#     gradient = cv2.filter2D(gradient, -1, kernel)
#     #Binarize the gradient
#     retval, binaryGrad = cv2.threshold(gradient.astype(np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     #Filter again to get rid of the noise of the binarization
#     binaryGradF = cv2.filter2D(binaryGrad, -1, kernel)
#     #Apply morphological operations
#     binaryGrad1 = cv2.morphologyEx(binaryGradF, cv2.MORPH_OPEN, kernel1)
#     binaryGrad2 = cv2.morphologyEx(binaryGradF, cv2.MORPH_CLOSE, kernel1)
#     binaryGrad3 = cv2.morphologyEx(binaryGrad1, cv2.MORPH_CLOSE, kernel1)
#     binaryGrad4 = cv2.morphologyEx(binaryGrad2, cv2.MORPH_OPEN, kernel1)
#     binaryGrad5 = cv2.morphologyEx(binaryGradF, cv2.MORPH_OPEN, kernel2)
#     binaryGrad6 = cv2.morphologyEx(binaryGradF, cv2.MORPH_CLOSE, kernel2)
#     binaryGrad5 = cv2.erode(binaryGrad5, kernel, iterations=2)
#     binaryGrad6 = cv2.dilate(binaryGrad6, kernel, iterations=2)
#
#     if plots:
#         plt.figure(),plt.imshow(binaryGrad1)
#         plt.figure(), plt.imshow(binaryGrad2)
#         plt.figure(), plt.imshow(binaryGrad3)
#         plt.figure(), plt.imshow(binaryGrad4)
#         plt.figure(), plt.imshow(binaryGrad5)
#         plt.figure(), plt.imshow(binaryGrad6)
#
#     im1 = binaryGrad1.astype(np.uint8)
#     im2 = binaryGrad2.astype(np.uint8)
#     im3 = binaryGrad3.astype(np.uint8)
#     im4 = binaryGrad4.astype(np.uint8)
#     im5 = binaryGrad5.astype(np.uint8)
#     im6 = binaryGrad6.astype(np.uint8)
#     #Compute point contours
#     contours1, thr = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours2, thr = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours3, thr = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours4, thr = cv2.findContours(im4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours5, thr = cv2.findContours(im5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours6, thr = cv2.findContours(im6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     #Create bboxes
#     x1, y1, w1, h1 = filterContours(contours1,imgGray,im1)
#     x2, y2, w2, h2 = filterContours(contours2, imgGray,im2)
#     x3, y3, w3, h3 = filterContours(contours3, imgGray,im3)
#     x4, y4, w4, h4 = filterContours(contours4, imgGray,im4)
#     x5, y5, w5, h5 = filterContours(contours5, imgGray,im5)
#     x6, y6, w6, h6 = filterContours(contours6, imgGray,im6)
#
#     if plots:
#         cv2.rectangle(imgGray, (x1, y1), (x1 + w1, y1 + h1), 255, 3)
#         plt.figure(), plt.imshow(imgGray)
#         cv2.rectangle(imgGray, (x2, y2), (x2 + w2, y2 + h2), 200, 3)
#         plt.figure(), plt.imshow(imgGray)
#         cv2.rectangle(imgGray, (x3, y3), (x3 + w3, y3 + h3), 150, 3)
#         plt.figure(), plt.imshow(imgGray)
#         cv2.rectangle(imgGray, (x4, y4), (x4 + w4, y4 + h4), 100, 3)
#         plt.figure(), plt.imshow(imgGray)
#         cv2.rectangle(imgGray, (x5, y5), (x5 + w5, y5 + h5), 70, 3)
#         plt.figure(), plt.imshow(imgGray)
#         cv2.rectangle(imgGray, (x6, y6), (x6 + w6, y6 + h6), 60, 3)
#         plt.figure(), plt.imshow(imgGray)
#
#     W = [w1,w2,w3,w4,w5,w6]
#     H = [h1,h2,h3,h4,h5,h6]
#     X = [x1,x2,x3,x4,x5,x6]
#     Y = [y1,y2,y3,y4,y5,y6]
#     ind = chooseBestBbox(W,H)
#     print('ind',ind)
#     iouSingle = iou([X[ind],Y[ind],X[ind]+W[ind],Y[ind]+H[ind]],gtBbox[i])
#     print(i)
#     print('iou',iouSingle)
#     allIou.append(iouSingle)
# meanIou = np.mean(allIou)
# print('MeanIoU:',meanIou)
#
#
