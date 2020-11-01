# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 08:20:24 2020

@author: mohaa
"""

import cv2 as cv
import glob


class imageReader(object):
    
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        
        self.imagePathList = self.getImagesPaths(self.path, self.extension)
        self.imgGrayArray, self.imgRGBArray = self.getImages(self.imagePathList)
    
    def getImagesPaths(self, path, extension):
        imgArray = []
        if extension == '.jpg':
            direction = glob.glob(str(path) + "\\*.jpg")
            for file in direction:
                imgArray.append(file)
                
        elif extension == '.png':
            for file in glob.glob(str(path) + "\\*.jpg"):
                imgArray.append(file)
        return imgArray
    
    def getImages(self, imgArray):
        imgGrayArray = []
        imgRGBArray = []
        
        for sample in imgArray:
            img = cv.imread(str(sample))
            imgRGBArray.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            imgGrayArray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
            
        return imgGrayArray, imgRGBArray