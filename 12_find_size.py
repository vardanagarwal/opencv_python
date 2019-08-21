# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 02:27:45 2019

@author: hp
"""

import cv2
import numpy as np

img = cv2.imread('Euro.JPG') 

image = img.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 255)# perform edge detection
_,cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finding external contours only
cv2.drawContours(img, cnts, -1, (0,255,0), 1)

for i in range(len(cnts)):
    approx = cv2.approxPolyDP(cnts[i], .03 * cv2.arcLength(cnts[i], True), True)
    if len(approx) == 8: # the value for circles is 8
        (x,y),radius = cv2.minEnclosingCircle(cnts[i]) # finds the minimum area circle
        center = (int(x), int(y))
        diameter = radius * 2
        #cv2.circle(img, center, int(radius), (0,255,0),2)

ratio = 23.25/diameter # size of euro is 23.25

for i in range(len(cnts)):
    approx = cv2.approxPolyDP(cnts[i], .03 * cv2.arcLength(cnts[i], True), True)
    if len(approx) == 4: # the value for rectangles is 4
        rect = cv2.minAreaRect(cnts[i]) # finds the minimum area rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(img,[box],0,(0,0,255),2)
        (p1, p2, p3, p4) = box # Unpacking tuple
        w = (((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5)*ratio # calculating width by calculating distance and multiplying by ratio
        h = (((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)**0.5)*ratio # calculating height by calculating distance and multiplying by ratio
        cv2.arrowedLine(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2, tipLength = 0.05)
        cv2.arrowedLine(img, (p2[0], p2[1]), (p1[0], p1[1]), (255, 0, 0), 2, tipLength = 0.05) # syntax repeated to obtain arrow on both sides
        cv2.arrowedLine(img, (p2[0], p2[1]), (p3[0], p3[1]), (0, 0, 255), 2, tipLength = 0.05)
        cv2.arrowedLine(img, (p3[0], p3[1]), (p2[0], p2[1]), (0, 0, 255), 2, tipLength = 0.05) # syntax repeated to obtain arrow on both sides
        cv2.putText(img, "{:.3f}mm".format(w), (int((p1[0]+p2[0])/2 - 20), int((p1[1]+p2[1])/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2) # putting side length on image
        cv2.putText(img, "{:.3f}mm".format(h), (int((p2[0]+p3[0])/2 - 20), int((p2[1]+p3[1])/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2) # putting side length on image

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()