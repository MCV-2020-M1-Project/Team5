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
        y, x, h, w = cv2.boundingRect(i)
        rects.append([x, y, w, h])
    rects = sorted(rects, key=rectangle_area, reverse=True)

    return rects


def inside_rectangle(rectangle_a, rectangle_b):

    # Return false if the position of one point of rectangle B is inside rectangle A.  Rectangle = [x,y,w,h]
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

    xa,ya,wa,ha = rectangle_b
    xb,yb,wb,hb = rectangle_a

    if xb>=xa and xb<=(xa+wa) and yb>=ya and yb<=(ya+ha): # Point xb,yb is inside A
        return True
    elif (xb+wb)>=xa and (xb+wb)<=(xa+wa) and yb>=ya and yb<=(ya+ha): # Point xb+wb,yb is inside A
        return True
    elif xb>=xa and xb<=(xa+wa) and (yb+hb)>=ya and (yb+hb)<=(ya+ha): # Point xb,yb+hb is inside A
        return True
    elif (xb+wb)>=xa and (xb+wb)<=(xa+wa) and (yb+hb)>=ya and (yb+hb)<=(ya+ha): # Point xb+wb,yb+hb is inside A
        return True

    return False

# Returns true if restrictions are satisfied
def satisfy_restrictions(rectangle, shape_image):

    min_prop_area = 0.2
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
    # edged = cv2.Canny(img_hsv_blur, 0, 255)
    edged = cv2.Canny(img_hsv_blur, 60, 120)
    # edged = cv2.Canny(img_hsv_blur, 80, 140)

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
        mask[x1:x1+w1, y1:y1+h1] = 1
        bbx.append([x1, y1, w1, h1])  # Save rectangle points
        if len(cnt2)>0:
            x2, y2, w2, h2 = cnt2
            mask[x2:x2 + w2, y2:y2 + h2] = 1
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
        if (w1*h1)<(1/10)*(np.size(image,0)*np.size(image,1)):
            x1 = 0
            y1 = 0
            w1 = np.size(image,0)
            h1 = np.size(image,1)
        # Search for a second painting
        found = False
        cnt1 = rects[0]
        rects = rects[1:]
        cnt2 = []
        cnt3 = []
        cmpt = 0
        while not found and (cmpt < len(rects)):
            if inside_rectangle([x1, y1, w1, h1], rects[cmpt]) or not satisfy_restrictions(rects[cmpt], image.shape):
                cmpt = cmpt + 1
            else:
                cnt2 = rects[cmpt]
                rects.pop(cmpt)
                found = True
        if len(cnt2) > 0:
            x2, y2, w2, h2 = cnt2
            cmpt = 0
            found = False
            while not found and (cmpt < len(rects)):
                if inside_rectangle([x1, y1, w1, h1], rects[cmpt]) or \
                        inside_rectangle([x2, y2, w2, h2], rects[cmpt]) or \
                        not satisfy_restrictions(rects[cmpt], image.shape):
                    cmpt = cmpt + 1
                else:
                    cnt3 = rects[cmpt]
                    found = True

        # Initialize mask & activate the pixels inside the rectangle
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[x1:x1 + w1,y1:y1 + h1] = 1
        if len(cnt2) > 0:
            x2, y2, w2, h2 = cnt2
            mask[x2:x2 + w2,y2:y2 + h2] = 1
        if len(cnt3) > 2:
            x3, y3, w3, h3 = cnt3
            mask[x3:x3 + w3, y3:y3 + h3] = 1

        sorted_paintings = orderPaintings(cnt1, cnt2, cnt3)
        for sp in sorted_paintings:
            cropped_images.append(image[sp[0]:sp[0]+sp[2],sp[1]:sp[1]+sp[3]])

    return cropped_images

def orderPaintings(cnt1,cnt2,cnt3):

    if cnt3:
        sortedcnt = sorted([cnt1,cnt2,cnt3], key=lambda x: x[1])
    elif cnt2:
        sortedcnt = sorted([cnt1, cnt2], key=lambda x: x[1])
    else:
        sortedcnt = [cnt1]

    return sortedcnt

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
