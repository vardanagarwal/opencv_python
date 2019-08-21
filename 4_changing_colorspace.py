# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:42:42 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('opencv.png')

def nothing(x):
    pass

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0])
upper = np.array([255,255,255])
mask = cv2.inRange(hsv, lower, upper)
extract = cv2.bitwise_and(img, img, mask = mask)

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R1','image',0,255,nothing)
cv2.createTrackbar('G1','image',0,255,nothing)
cv2.createTrackbar('B1','image',0,255,nothing)
cv2.createTrackbar('R2','image',0,255,nothing)
cv2.createTrackbar('G2','image',0,255,nothing)
cv2.createTrackbar('B2','image',0,255,nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',extract)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    mask = cv2.inRange(hsv, lower, upper)
    extract = cv2.bitwise_and(img, img, mask = mask)
    # get current positions of four trackbars
    r1 = cv2.getTrackbarPos('R1','image')
    g1 = cv2.getTrackbarPos('G1','image')
    b1 = cv2.getTrackbarPos('B1','image')
    r2 = cv2.getTrackbarPos('R2','image')
    g2 = cv2.getTrackbarPos('G2','image')
    b2 = cv2.getTrackbarPos('B2','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        mask[:] = 0
    else:
        l = np.uint8([[[r1, b1, g1]]])
        u = np.uint8([[[r2, b2, g2]]])
        lower = cv2.cvtColor(l, cv2.COLOR_BGR2HSV)
        upper = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)

cv2.destroyAllWindows()