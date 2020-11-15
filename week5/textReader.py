import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

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

        cropped_text = pytesseract.image_to_string(cropped_image)
        opening_text = pytesseract.image_to_string(opening)
        closing_text = pytesseract.image_to_string(closing)

        # remove unnecessary characters
        cropped_text = cropped_text.replace('\n', '')
        cropped_text = cropped_text.replace('\f', '')

        opening_text = opening_text.replace('\n', '')
        opening_text = opening_text.replace('\f', '')

        closing_text = closing_text.replace('\n', '')
        closing_text = closing_text.replace('\f', '')

        self.cropped_text = cropped_text
        self.opening_text = opening_text
        self.closing_text = closing_text