import numpy as np
import cv2
from skimage.transform import probabilistic_hough_line

def rotate_imgs(query):
    rotated_images = []
    angles = []
    for i in range(0,len(query)):

        img = query[i]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        # cv2.Canny(image, intensity_H_threshold, intensity_L_threshold, apertureSize, L2gradient)
        edges = cv2.Canny(gray_blur, 50, 200)
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=1)

        # cv2.HoughLinesP(edges, rho = 1, theta = math.pi / 180, threshold = 70, minLineLength = 100, maxLineGap = 10)
        lines = probabilistic_hough_line(dilation, threshold=300, line_length=int(np.floor(np.size(img, 0) / 5)),line_gap=3)
        max = 0
        angle = 0
        pos = 0
        if lines != []:
            for j in range(0, len(lines)):
                line = lines[j]
                # cv2 and skimage create the lines the opposite ways
                x1, y1, x2, y2 = correcCoordSkimage(line)
                val = x1 + y1
                angle_aux = getAngle(x1, y1, x2, y2)
                if (val > max or max == 0) and (angle_aux < 45 and angle_aux > -45):
                    angle = angle_aux
                    max = val
                    pos = j
            x1, y1, x2, y2 = correcCoordSkimage(lines[pos])
            #cv2.line(img, (x1, y1), (x2, y2), (125), 5)    # this codeline draws the line to the img

        #Correct the angles
        if angle > 45:
            angle = 90 - angle
        elif angle < -45:
            angle = angle + 90

        img = rotate(img, angle)
        rotated_images.append(img)

        # Correct the angles
        if angle < 0:
            angle = angle + 180
        angles.append(angle)
    return rotated_images, angles

def getAngle(x1,y1,x2,y2):
    if x2 == x1:
        angle = 90
    elif y1 == y2:
        angle = 0
    else:
        angle = np.arctan((y2 - y1) / (x2 - x1))
        # transform to degrees
        angle = angle * 180 / np.pi
    return -angle   # anti-clockwise

def rotate(img, angle):
    h = int(round(np.size(img,0)/2))
    w = int(round(np.size(img,1)/2))
    center = (w,h)
    M = cv2.getRotationMatrix2D(center, -angle, 1)
    result = cv2.warpAffine(img, M, (w*2, h*2))
    return result

def correcCoordSkimage(line):
    x2 = line[0][0]
    y2 = line[0][1]
    x1 = line[1][0]
    y1 = line[1][1]
    if x1 > x2:
        temp = x1
        x1 = x2
        x2 = temp
        temp = y1
        y1 = y2
        y2 = temp
    return x1,y1,x2,y2
