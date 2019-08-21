# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 03:42:48 2019

@author: hp
"""

import cv2
import numpy as np
img = cv2.imread('sudoku_processed.png', 0)
kernel = np.ones((5, 5), np.uint8)

#erosion
erosion = cv2.erode(img, kernel, iterations = 1)
#dilation
dilation = cv2.dilate(img, kernel, iterations = 1)
#opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#top hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#black hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
kernel2 = np.ones((1,1), np.uint8)
blackhat = cv2.erode(blackhat, kernel2, iterations = 1)
kernel2 = np.ones((1,1), np.uint8)
blackhat = cv2.dilate(blackhat, kernel2, iterations = 1)
ret, blackhat = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
blackhat = cv2.bitwise_not(blackhat)

images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]
titles = ['original', 'erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']
for i in range(0,8):
    cv2.imshow(titles[i], images[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()