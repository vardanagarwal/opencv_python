# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 03:19:59 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('opencv.png')

#filtering
kernel = np.ones((5,5), np.float32)/25 # custom kernel
filtered = cv2.filter2D(img, -1, kernel) # for filtering of custom kernel

#smoothing
#averaging
blur = cv2.blur(img, (5,5))
#gaussian blur
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0) # effective against Gaussian noise
#median blur
median_blur = cv2.medianBlur(img, 5) # effective against salt and pepper noise
#bilateral filtering
bilateral_filtering = cv2.bilateralFilter(img, 9, 75, 75)

titles = ['Original', 'Filtered', 
          'Averaging', 'Gaussian',
          'Median', 'Bilateral filtering']
images = [img, filtered, blur, gaussian_blur, median_blur, bilateral_filtering]
for i in range(6):
    cv2.imshow(titles[i], images[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()