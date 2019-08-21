# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:26:34 2019

@author: hp
"""
from __future__ import print_function
import cv2
import numpy as np

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

#Perspective Transformation
img2 = cv2.imread('sudoku.jpg')
#img2 = cv2.medianBlur(img2,5)
#rows,cols,ch = img2.shape
img2 = cv2.resize(img2, (500, 500),  interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
dst = cv2.fastNlMeansDenoising(gray,None,10,7,21)
ret, mask = cv2.threshold(dst, 45, 255, cv2.THRESH_BINARY)
#mask = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\cv2.THRESH_BINARY,11,2)
canny = cv2.Canny(mask, 135, 255)
kernel = np.ones((3, 3),np.uint8)
canny = cv2.dilate(canny,kernel,iterations = 1)
im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img2, contours, -1, (0,255,0), 1)
cnt = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(cnt)
#cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,255), 2)
#hull = cv2.convexHull(cnt,returnPoints = False)
#defects = cv2.convexityDefects(cnt,hull)
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
#cv2.drawContours(img2, [hull], -1, (255, 255, 255), -1)
#for i in range(defects.shape[0]):
#    s,e,f,d = defects[i,0]
#    start = tuple(cnt[s][0])
#    end = tuple(cnt[e][0])
#    far = tuple(cnt[f][0])
#    cv2.line(img2,start,end,[0,255,0],2)
#    cv2.circle(img2,far,5,[0,0,255],-1)
#pts1 = np.float32([[rect[0,0],rect[0,1]],[rect[1,0],rect[1,1]],[rect[2,0],rect[2,1]],[rect[2,0],rect[2,1]]])
#pts2 = np.float32([[64,75],[439,60],[466,463],[30,460]])
M = cv2.getPerspectiveTransform(rect, dest)
intermediate = cv2.warpPerspective(img2,M,(maxWidth, maxHeight))
#cv2.imshow('perspective transform', intermediate)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#intermediate preprocessing

gamma = 0.8
invGamma = 1/gamma
table = np.array([((i / 255.0) ** invGamma) * 255
                  for i in np.arange(0, 256)]).astype("uint8")
cv2.LUT(intermediate, table, intermediate)
    
img = cv2.GaussianBlur(intermediate,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
#kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
kernel1 = np.ones((11, 11), np.uint8)
close = cv2.dilate(gray, kernel1, iterations = 1)
#close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res1 = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res1,cv2.COLOR_GRAY2BGR)


#Finding vertical lines
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
dx = cv2.Sobel(res1,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

im2, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

#finding horizontal lines
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res1, cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely, iterations = 1)

im2, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()


res = cv2.bitwise_and(closex,closey)
im2, contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img2,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))
centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((100,2))
c2 = c[np.argsort(c[:,1])]

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
bm = b.reshape((10,10,2))

output = np.zeros((450,450),np.uint8)
for i,j in enumerate(b):
    ri = int(i/10)
    ci = int(i%10)
    if ci != 9 and ri!=9:
        src = bm[int(ri):int(ri+2), int(ci):int(ci+2) , :].reshape((4,2))
        dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32)
        retval = cv2.getPerspectiveTransform(src,dst)
        warp = cv2.warpPerspective(res1 ,retval,(450,450))
        output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()
cv2.imshow('final', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('sudoku_processed.png', output)
