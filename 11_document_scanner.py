# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 02:16:25 2019

@author: hp
"""

import cv2
import numpy as np
from skimage.filters import threshold_local

img = cv2.imread('page.jpg')
blur = cv2.GaussianBlur(img,(11,11),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
canny = cv2.Canny(mask, 30, 200)
(_, cnts, _) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        #cv2.drawContours(img, [approx], -1, (0, 0, 255), 1)
        cnt = c
        break
        
#return the four points required for perspective transform
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

hull = cv2.convexHull(cnt)
rect = order_points(hull[:,0,:])
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
dest = np.array([[0,0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect, dest)
scanned = cv2.warpPerspective(img,M,(maxWidth, maxHeight))

#gamma = 2
#invGamma = 1/gamma
#table = np.array([((i / 255.0) ** invGamma) * 255
#                  for i in np.arange(0, 256)]).astype("uint8")
#cv2.LUT(scanned, table, scanned)

#lab= cv2.cvtColor(scanned, cv2.COLOR_BGR2LAB)
#l, a, b = cv2.split(lab)
#clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#cl = clahe.apply(l)
#limg = cv2.merge((cl,a,b))
#final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


#scanned = cv2.GaussianBlur(scanned,(5,5),0)
gray = cv2.cvtColor(scanned,cv2.COLOR_BGR2GRAY)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#gray = clahe.apply(gray)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#kernel2 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
gray2 = cv2.filter2D(gray, -1, kernel) # sharpening using kernel

T = threshold_local(gray, 11, offset = 10, method = "gaussian")
mask1 = (gray > T).astype("uint8") * 255
T = threshold_local(gray2, 11, offset = 10, method = "gaussian")
mask2 = (gray2 > T).astype("uint8") * 255
mask = cv2.bitwise_or(mask1, mask2)

cv2.imshow('first', mask1)
cv2.imshow('second', mask2)
cv2.imshow('result', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()