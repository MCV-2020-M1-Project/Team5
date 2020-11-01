#Quan tenim coord bbox → binaritzar per diferenciar les lletres → aplicar OCR (linia codi a les slides).

#Mapejar nom extret amb nom de pintor a la bbdd.
#Mètriques: Levenshtein...https://www.kdnuggets.com/2019/01/comparison-text-distance-metrics.html

#Combinant amb el punt anterior: test retrieval usant els descriptors de color de la W2

from utils import *
import cv2
import pickle
import numpy as np
from imageReader import imageReader as imgReader
import pytesseract


class textReader():
    
    def __init__(self, image, bBox):
        
        self.image = image
        self.bBox = bBox
        
        x1 = self.bBox[0]
        x2 = self.bBox[2]
        y1 = self.bBox[1]
        y2 = self.bBox[3]
        w = self.bBox[2] - x1
        h = self.bBox[3] - y1

        cropped_image = self.image[y1:y2, x1:x2]
        
        # Make opening and closing to focus the text
        kernel = np.ones((3,1),np.uint8)
        opening = cv2.morphologyEx(cropped_image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, kernel) 
        
        # binarize those images
        ret, openingBin = cv2.threshold(opening,110,255,cv2.THRESH_BINARY)
        ret, closingBin = cv2.threshold(closing,110,255,cv2.THRESH_BINARY)
        
        #cv2.imshow("Original image",self.image)
        #cv2.imshow("Croppen image",cropped_image)
        #cv2.imshow("Opening",opening)
        #cv2.imshow("Closing",closing)
        #cv2.imshow("OpeningBin",openingBin)
        #cv2.imshow("ClosingBin",closingBin)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        opening_text = pytesseract.image_to_string(cropped_image)
        closing_text = pytesseract.image_to_string(cropped_image)

        resulting_text = opening_text.replace('\n', '')
        resulting_text = resulting_text.replace('\f', '')

        self.opening_text = resulting_text
