# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:52:31 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('opencv.png')
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 4) 
# image name, start point, end point, color, thickness

cv2.rectangle(img,(384, 0), (510, 128), (0, 255, 0), 3)
# image name, top corner points, bottom corner points, color, thickness

cv2.circle(img,(447, 63), 63, (127, 128, 25), -1)
# image name, centre, radius, color, thickness
# -1 denotes filled 

cv2.ellipse(img, (256, 256), (100, 50), 45, 0, 180, (255, 0, 0), -1)
# image name, centre, major and minor axis length, angle of rotation of ellipse, start angle, end angle, color, thickness 
#if start angle is 0 and end angle is 360 it denotes full eclipse

pts = np.array([[10, 5], [200, 200], [500, 200], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2)) # need to reshape points as rows X 1 X 2
cv2.polylines(img, [pts], True, (0, 0, 0), 2) # to make a polygon
# image name, points, True or False, Color, thickness
#if True then a closed polygon is formed, if false the first and last points are not connected

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (25, 255, 255), 2, cv2.LINE_AA) 
# image name, Text to put, where to start text, font type, font scale, color, thickness, line type

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#TRACKBAR
import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image') # creates an output window

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
# trackbar name, output window name, start, end, function

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image') # trackbar name, output window name
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()