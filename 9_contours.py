# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 04:36:53 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('shapes.png')
height, width = img.shape[:2]
M = cv2.getRotationMatrix2D((0, 0), 15, 1)#params- centre, angle, scale
img = cv2.warpAffine(img, M, (width, height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
im, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)
M = cv2.moments(cnts[1])
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
centroid = (cx, cy)
cv2.circle(img,centroid, 5, (127, 128, 25), -1)
x, y, w, h = cv2.boundingRect(cnts[3])
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
rect = cv2.minAreaRect(cnts[3])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (255, 0, 255), 2)
(x, y),(mA, ma), angle = cv2.fitEllipse(cnts[2])
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()