# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 02:10:28 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('lena.jpg')
height = img.shape[0]
width = img.shape[1]
m = max(height, width)
i = np.zeros((m*4, m*4, 3), dtype = 'uint8')
i[int(3*m/2) : int(3*m/2+height), int(3*m/2) : int(3*m/2+width)] = img
M = cv2.getRotationMatrix2D((3*m/2, 3*m/2), 45, 1)#params- centre, angle, scale(in this image rotated at origin of original image 
#for centre use 2*width and 2*height)
rotated = cv2.warpAffine(i, M, (m*4, m*4))
img2gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(rotated, contours, -1, (0,255,0), 3)
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
wanted_part = rotated[y:y+h, x:x+w]
#cv2.rectangle(rotated,(x,y),(x+w,y+h),(0,255,255),2)
cv2.imshow("Rotate", wanted_part)
cv2.waitKey(0)
cv2.destroyAllWindows()