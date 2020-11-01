import cv2
from evaluation import *
import pickle


def rectangle_area(rect):
    x, y, w, h = rect
    return w*h


def contour2rectangle(contours):
    # Get bounding rectangle for each found contour and sort them by area
    rects = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        rects.append([x, y, w, h])
    rects = sorted(rects, key=rectangle_area, reverse=True)

    return rects


def inside_rectangle(rectangle_a, rectangle_b):

    # Return false if the position of one point of rectangle B is inside rectangle A.  Rectangle = [x,y,w,h]
    is_inside = False
    xa,ya,wa,ha = rectangle_a
    xb,yb,wb,hb = rectangle_b
    if xb>=xa and xb<=(xa+wa) and yb>=ya and yb<=(ya+ha): # Point xb,yb is inside A
        return True
    elif (xb+wb)>=xa and (xb+wb)<=(xa+wa) and yb>=ya and yb<=(ya+ha): # Point xb+wb,yb is inside A
        return True
    elif xb>=xa and xb<=(xa+wa) and (yb+hb)>=ya and (yb+hb)<=(ya+ha): # Point xb,yb+hb is inside A
        return True
    elif (xb+wb)>=xa and (xb+wb)<=(xa+wa) and (yb+hb)>=ya and (yb+hb)<=(ya+ha): # Point xb+wb,yb+hb is inside A
        return True
    else:
        return is_inside


# Returns true if restrictions are satisfied
def satisfy_restrictions(rectangle, shape_image):

    min_prop_area = 0.01
    min_ratio = 0.25
    max_ratio = 4
    x, y, w, h = rectangle

    # rect has a minimum area
    if w * h < (shape_image[0]*min_prop_area)*(shape_image[1]*min_prop_area):
        return False

    # ratio of h/w isn't smaller than 1/4
    ratio = w / h
    if ratio <= min_ratio or ratio >= max_ratio:
        return False

    return True


def compute_contours(image):

    # Convert image to HSV an use only saturation channel (has most information)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # We apply an gaussian filter to remove possible noise from the image
    img_hsv_blur = cv2.GaussianBlur(img_hsv[:, :, 1], (5, 5), 0)

    # Get edges using Canny algorithm
    edged = cv2.Canny(img_hsv_blur, 0, 255)

    # Apply close transformation to eliminate smaller regions
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)/v.CHAIN_APPROX_SIMPLE

    return contours


# Subtracts de background from the image and returns cropped_img and mask (mask = rectangle)
def compute_mask(image):

    contours = compute_contours(image)
    bbx = []

    # If contours not found, pass whole image
    if not contours:
        mask = np.ones(image.shape)
    else:
        rects = contour2rectangle(contours)
        x1,y1,w1,h1 = rects[0]

        # Search for a second painting
        found = False
        rects = rects[1:]
        cnt2 = []
        cmpt = 0
        while not found and (cmpt<len(rects)):
            if inside_rectangle([x1, y1, w1, h1], rects[cmpt]) or not satisfy_restrictions(rects[cmpt],image.shape):
                cmpt = cmpt+1
            else:
                cnt2 = rects[cmpt]
                found = True

        # Initialize mask & activate the pixels inside the rectangle
        mask = np.zeros(image.shape[:2],np.uint8)
        mask[y1:y1+h1, x1:x1+w1] = 1
        bbx.append([x1, y1, w1, h1])  # Save rectangle points
        if len(cnt2)>0:
            x2, y2, w2, h2 = cnt2
            mask[y2:y2 + h2, x2:x2 + w2] = 1
            if x2<x1 or y2<y1: # Order rectangles from left to right or top to bottom
                bbx = [[x2, y2, w2, h2], [x1, y1, w1, h1]]
            else:
                bbx.append([x2, y2, w2, h2])  # Save rectangle points

    # Mask multiplied *255 to equal the values of the groundtruth images
    return bbx, mask*255


# Subtracts de background from the image and returns cropped_img and mask (mask = rectangle)
def compute_croppedimg(image):

    contours = compute_contours(image)
    cropped_images = []

    # If contours not found, pass whole image
    if not contours:
        cropped_images.append(image)
    else:
        rects = contour2rectangle(contours)
        x1, y1, w1, h1 = rects[0]

        # Search for a second painting
        found = False
        rects = rects[1:]
        cnt2 = []
        cmpt = 0
        while not found and (cmpt < len(rects)):
            if inside_rectangle([x1, y1, w1, h1], rects[cmpt]) or not satisfy_restrictions(rects[cmpt], image.shape):
                cmpt = cmpt + 1
            else:
                cnt2 = rects[cmpt]
                found = True

        # Initialize mask & activate the pixels inside the rectangle
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[y1:y1 + h1, x1:x1 + w1] = 1
        cropped_images.append(image[y1:y1 + h1, x1:x1 + w1, :])
        if len(cnt2) > 0:
            x2, y2, w2, h2 = cnt2
            mask[y2:y2 + h2, x2:x2 + w2] = 1
            if x2 < x1 or y2 < y1:  # Order rectangles from left to right or top to bottom
                cropped_images = [image[y2:y2 + h2, x2:x2 + w2, :], image[y1:y1 + h1, x1:x1 + w1, :]]
            else:
                cropped_images.append(image[y2:y2 + h2, x2:x2 + w2, :])  # Save rectangle points

    return cropped_images


# Subtracts de background from a list of images and returns a list of masks
def get_mask_array(imgarray):

    masks = []
    for i in imgarray:
        masks.append(compute_mask(i)[1])

    return masks


# Gets boundingboxes from the pictures in the images (Max: 2 pictures/image)
def get_pictbbs_array(imgarray):

    masks = []
    for i in imgarray:
        masks.append(compute_mask(i)[0])

    return masks


# returns cropped images, separated paintings not images
def crop_imgarray(imgarray):

    cropped_imgs = []
    for i in imgarray:
        paints_img = []
        crpimg = compute_croppedimg(i)
        for painting in crpimg:
            paints_img.append(painting)
        cropped_imgs.append(paints_img)

    return cropped_imgs


# Return list of images with list of paintings, and also saves it if given a filename
def get_result_pkl(imgarray,filename=None):

    cropped_imgs = []
    for i in imgarray:
        cropped_imgs.append(compute_croppedimg(i))

    if filename:
        outfile = open(filename,'wb')
        pickle.dump(cropped_imgs,outfile)
        outfile.close()

    return cropped_imgs


def mask_evaluation(images,masks):

    PRs = []
    RCs = []
    F1s = []

    for i in range(len(images)):
        PR, RC, F1 = evaluation(images[i][:,:,0], masks[i])
        PRs.append(PR)
        RCs.append(RC)
        F1s.append(F1)

    return PRs, RCs, F1s
