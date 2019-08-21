# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 05:28:34 2019

@author: hp
"""

import cv2 
import numpy as np
img = cv2.imread('messi.jpg')
height, width = img.shape[:2]

#Scaling (interpolation - for zooming inter_cubic preferred and for shrinking inter_area preferred)
zoom = cv2.resize(img, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
shrink = cv2.resize(img, (int(width/2), int(height/2)), interpolation = cv2.INTER_AREA)
#both the above methods can be used
cv2.imshow("Original", img)
cv2.imshow("Zoom", zoom)
cv2.imshow("Shrink", shrink)
cv2.waitKey(0)
cv2.destroyAllWindows()

#translation
M = np.float32([[1, 0, 10], [0, 1, 20]])#use [[1,0,-10],[0, 1,-20]] for other sides. - on third argument.
translated = cv2.warpAffine(zoom, M, (width*4, height*4))
cv2.imshow("Original", zoom)
cv2.imshow("Translate", translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#rotation
M = cv2.getRotationMatrix2D((0, 0), 45, 1)#params- centre, angle, scale
rotated = cv2.warpAffine(img, M, (width, height))
cv2.imshow("Rotate", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#affine transform
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(zoom, M, (width*4, height*4))
cv2.imshow('affine transform', affine)
cv2.waitKey(0)
cv2.destroyAllWindows()

#perspective transforms example will be shown later as it uses some other methods as well which are not disscussd right now.