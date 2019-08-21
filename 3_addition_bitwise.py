# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:11:47 2019

@author: hp
"""

import cv2

img1 = cv2.imread('lena.jpg')
img2 = cv2.imread('baboon.jpg')

dst1 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
# image 1, alpha, image 2, beta, gamma
# final result = alpha*image1 + beta*image2 + gamma
dst2 = cv2.add(img1, img2) # this simply add the pixels
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# bitwise operations can only be performed on binary images. To obtain binary images we perform thresholding.
img1 = cv2.imread('opencv.png')
img2 = cv2.imread('blackonwhite.jpg')
rows, col, channels = img2.shape # returns size of image
roi = img1[0:rows, 0:col]
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) # thresholding will be done in detail later
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
img2_fg = cv2.bitwise_and(img2, img2, mask = mask_inv)#if white on black is used then invert the masks i.e. for the step above use mask_inv and for this use mask
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:col] = dst;
cv2.imshow('Image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()