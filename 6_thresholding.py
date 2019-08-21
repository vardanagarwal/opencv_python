# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:48:43 2019

@author: hp
"""

import cv2

#simple thresholdings
img = cv2.imread('race_track.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    cv2.imshow(titles[i], images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()

#adaptive thresholdings and Otsu's thresholding
img = cv2.imread('race_track.png',0)
img = cv2.medianBlur(img,5)
img = cv2.resize(img, (500, 500),  interpolation = cv2.INTER_CUBIC)
img = cv2.fastNlMeansDenoising(img,None,10,7,21)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

ret4, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'OTSU Thresholding']
images = [img, th1, th2, th3, th4]

for i in range(5):
    cv2.imshow(titles[i], images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()