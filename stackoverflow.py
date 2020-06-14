# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 18:40:23 2019

@author: hp
"""
import cv2
import numpy as np

#%%
img = cv2.imread('some_mask.jpg',0)
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if img[i,j] == 0:
            print(i,j)
                      
zoom = cv2.resize(img, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
ret,thresh1 = cv2.threshold(zoom, 16, 255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(zoom, 17, 255, cv2.THRESH_BINARY)
output = cv2.bitwise_xor(thresh1, thresh2)
cv2.imshow('thresh=16', thresh1)
cv2.imshow('thresh>16', thresh2)
cv2.imshow('result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('oath.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
ret, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('original', mask)
hist = cv2.equalizeHist(denoise)
gamma = 0.5
invGamma = 1/gamma
table = np.array([((i / 255.0) ** invGamma) * 255
                  for i in np.arange(0, 256)]).astype("uint8")
gamm = cv2.LUT(denoise, table, denoise)
blur = cv2.GaussianBlur(gamm,(5,5),0)
cv2.imshow('modified', gamm)
mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#canny = cv2.Canny(mask, 40, 200)
cv2.imshow('result', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img1 = cv2.imread('baboon.jpg')
img2 = cv2.imread('lena.jpg')
def MSE(img1, img2):
        squared_diff = img1 -img2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return err
MSE(img1,img2)    


#%%
small = cv2.imread('small_find.png')
large = cv2.imread('large_find.jpg')
pixel = np.reshape(small[3,3], (1,3))
lower =[pixel[0,0]-10,pixel[0,1]-10,pixel[0,2]-10]
lower = np.array(lower, dtype = 'uint8')
upper =[pixel[0,0]+10,pixel[0,1]+10,pixel[0,2]+10]
upper = np.array(upper, dtype = 'uint8')
mask = cv2.inRange(large,lower, upper)
mask2 = cv2.inRange(small, lower, upper)
im, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(large, contours, -1, (0,0,255), 1)
cnt = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(cnt)
wanted_part = mask[y:y+h, x:x+w]
wanted_part = cv2.resize(wanted_part, (mask2.shape[1], mask2.shape[0]), interpolation = cv2.INTER_LINEAR)
cv2.imshow('image1', wanted_part)
cv2.imshow('image', mask2)
MSE(wanted_part, mask2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output.png", np.hstack([cv2.bitwise_not(mask2), cv2.bitwise_not(wanted_part)]))

#%%
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

#%%
img = cv2.imread('contouring.png',0)
bgr = np.ones((img.shape[0], img.shape[1], 3), dtype= 'uint8')*255
_,contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('image1', img)
cv2.imshow('image2', bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
bgr = np.zeros((img.shape[0], img.shape[1]), dtype= 'uint8')*255
cv2.drawContours(bgr, contours, -1, (255,255,255), 1)
bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8), iterations=1)
bgr = cv2.bitwise_and(bgr, cv2.bitwise_not(img))

#%%
img = cv2.imread('lena.jpg')
height = img.shape[0]
width = img.shape[1]
m = max(height, width)
i = np.zeros((m*4, m*4, 3), dtype = 'uint8')
i[int(3*m/2) : int(3*m/2+height), int(3*m/2) : int(3*m/2+width)] = img
angle = 30
M = cv2.getRotationMatrix2D((3*m/2, 3*m/2),angle , 1)#params- centre, angle, scale(in this image rotated at origin of original image 
#for centre use 2*width and 2*height)
rotated = cv2.warpAffine(i, M, (m*4, m*4))
img2gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(rotated, contours, -1, (0,255,0), 3)
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
wanted_part = rotated[y:y+h, x:x+w]
#cv2.rectangle(rotated,(x,y),(x+w,y+h),(0,255,255),2)
cv2.imshow("Rotate", wanted_part)
cv2.waitKey(0)
cv2.destroyAllWindows()

import math
x1,y1 = 50,50
dist1 = (x1-0)**2 +(y1-0)**2
dist2 = (x1-width/2)**2+(y1-height/2)**2
origin_y = round(math.sin(math.radians(angle))*width)
origin_x = 0
centre_y = wanted_part.shape[0]/2
centre_x = wanted_part.shape[1]/2

from sympy.solvers import solve
from sympy.abc import x,y
solve([(x-origin_x)**2 + (y-origin_y)**2 - dist1,(x-origin_x)**2 + (y-origin_y)**2 - dist1],x,y )


#%%
img = cv2.imread('lines.png')
kernel = np.array([[0, -1, 0],
                   [1, 0, 1],
                   [0, -1, 0]])

dst = cv2.filter2D(img, -1, kernel)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(dst, kernel, iterations = 1)
kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
_,thresh = cv2.threshold(opening,10,255,cv2.THRESH_BINARY)
res = cv2.bitwise_and(img, cv2.bitwise_not(thresh))
kernel = np.ones((7, 7), np.uint8)
other = cv2.erode(img, kernel, iterations = 1)
_,thresh = cv2.threshold(other,10,255,cv2.THRESH_BINARY)
result = cv2.bitwise_and(res, cv2.bitwise_not(thresh))
cv2.imshow("Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#img = cv2.imread('artificial_stroke.jpg')
#img = img[300:600,250:762,:]
#img1 = img
#cv2.namedWindow('image')
#def nothing(x):
#    pass
#cv2.createTrackbar('1','image',1,11,nothing)
#cv2.createTrackbar('2','image',1,11,nothing)
#cv2.createTrackbar('3','image',1,11,nothing)
##cv2.createTrackbar('4','image',-10,10,nothing)
##cv2.createTrackbar('5','image',-10,10,nothing)
##cv2.createTrackbar('6','image',-10,10,nothing)
##cv2.createTrackbar('7','image',-10,10,nothing)
##cv2.createTrackbar('8','image',-10,10,nothing)
##cv2.createTrackbar('9','image',-10,10,nothing)
#switch = '0 : OFF \n1 : ON'
#cv2.createTrackbar(switch, 'image',0,1,nothing)
#
#while(1):
#    cv2.imshow('image',img1)
#    k = cv2.waitKey(500)
#    if k == 27:
#        break
#
#    r1 = cv2.getTrackbarPos('1','image')
#    r2 = cv2.getTrackbarPos('2','image')
#    r3 = cv2.getTrackbarPos('3','image')
#    if r1<6:
#        r1 = r1-10 + (r1-1)*r1
#    elif r1 == 6:
#        r1 = 0
#    else:
#        r1 = 2*r1 -13
#    if r2<6:
#        r2 = r2-10 + (r2-1)*r2
#    elif r2 == 6:
#        r2 = 0
#    else:
#        r2 = 2*r2 -13
#    if r3<6:
#        r3 = r3-10 + (r3-1)*r3
#    elif r3 == 6:
#        r3 = 0
#    else:
#        r3 = 2*r3 -13
##    r4 = cv2.getTrackbarPos('4','image')
##    r5 = cv2.getTrackbarPos('5','image')
##    r6 = cv2.getTrackbarPos('6','image')
##    r7 = cv2.getTrackbarPos('7','image')
##    r8 = cv2.getTrackbarPos('8','image')
##    r9 = cv2.getTrackbarPos('9','image')
#    s = cv2.getTrackbarPos(switch,'image')
#
#    if s==0:
#        img1 = img
#    else:
#        kernel = np.array([[r1, r2, r1],
#                   [r2, r3, r2],
#                   [r1, r2, r1]])
#        img1 = cv2.filter2D(img, -1, kernel)
#
#cv2.destroyAllWindows()


#%%
original = cv2.imread('inner.png')
img = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
ret,thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
thresh1 = cv2.Canny(thresh1, 10, 255)
thresh2 = cv2.Canny(thresh2, 10, 255)
im, contours1, _ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im, contours2, _ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(original, contours1, -1, (0, 255, 255), 3)
cv2.drawContours(original, contours2, -1, (0, 255, 255), 3)
cv2.imshow('image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('red_green.png')
red = img[:, :, 2] # to segment out red area
green = img[:, :, 1] # to segment out green are
ret, thresh1 = cv2.threshold(red, 5, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(green, 5, 255, cv2.THRESH_BINARY)
_, cnts1, _ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
_, cnts2, _ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c1 = max(cnts1, key = cv2.contourArea)
c2 = max(cnts2, key = cv2.contourArea)
rect1 = cv2.minAreaRect(c1)
rect2 = cv2.minAreaRect(c2)
box1 = cv2.boxPoints(rect1)
box2 = cv2.boxPoints(rect2)
box1 = np.int0(box1)
box2 = np.int0(box2)
cv2.drawContours(img, [box1], 0, (0, 255, 255), 2)
cv2.drawContours(img, [box2], 0, (0, 255, 255), 2)
(p1, p2, p3, p4) = box1 # Unpacking tuple
h1 = (((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5) # calculating width by calculating distance
w1 = (((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)**0.5) # calculating height by calculating distance
(p1, p2, p3, p4) = box2 # Unpacking tuple
h2 = (((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5) # calculating width by calculating distance
w2 = (((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)**0.5) # calculating height by calculating distance
rofh = h2/h1
rofw = w2/w1
print("ratio of height = ", rofh, "and ratio by width = ", rofw)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('cloud.png', 0)
height = img.shape[0]
width = img.shape[1]
ret, thresh = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY)
total = sum(map(sum, thresh))
percent = total/height/width*100
print('percentage of cloud cover is =', percent, '%')
cv2.imshow('image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('grayed.png', 0)
ret,thresh = cv2.threshold(img,64,255,cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
erode = cv2.erode(thresh, kernel, iterations = 1)
result = cv2.bitwise_or(img, erode)
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('ink_mark.png')
wimg = img[:, :, 0]
ret,thresh = cv2.threshold(wimg,100,255,cv2.THRESH_BINARY)

kernel = np.ones((7, 7), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
erosion = cv2.erode(closing, kernel, iterations = 1)
mask = cv2.bitwise_or(erosion, thresh)
white = np.ones(img.shape,np.uint8)*255
white[:, :, 0] = mask
white[:, :, 1] = mask
white[:, :, 2] = mask
result = cv2.bitwise_or(img, white)

cv2.imshow('image', wimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
def find_nn(point, neighborhood):
    """
    Finds the nearest neighborhood of a vector.

    Args:
        point (float array): The initial point.
        neighborhood (numpy float matrix): The points that are around the initial point.

    Returns:
        float array: The point that is the nearest neighbor of the initial point.
        integer: Index of the nearest neighbor inside the neighborhood list
    """
    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point, neighbor, cv2.NORM_L2)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i
    nn_idx = nn_idx + j + 1
    return nn, nn_idx 

#taking 6 random points on a board of 200 x 200
points = [(10, 10), (115, 42), (36, 98), (78, 154), (167, 141), (189, 4)]
board = np.ones((200, 200, 3), dtype = np.uint8) * 255
for i in range(6):
    cv2.circle(board, points[i], 5, (0, 255, 255), -1)

for j in range(5):
    nn, nn_idx = find_nn(points[j], points[j+1:])
    points[j+1], points[nn_idx] = points[nn_idx], points[j+1]
    
for i in range(5):
    cv2.arrowedLine(board, points[i], points[i+1], (255, 0, 0), 1, tipLength = 0.07)
cv2.imshow('image', board)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('vertical_noise.jpg')
img = cv2.GaussianBlur(img, (11, 1), 0)
#img = cv2.blur(img,(15, 1))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import dlib
from imutils.face_utils import shape_to_np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('res/model.dat')

face = cv2.imread('face.jpg', 0)
face = cv2.resize(face, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
img = face.copy()
dets = detector(img, 0)
for i, det in enumerate(dets):
    shape = shape_to_np(predictor(img, det))
    shape_left_eye = shape[36:42]
    x, y, w, h = cv2.boundingRect(shape_left_eye)
#    cv2.rectangle(face, (x, y), (x + w, y + h), (255, 255, 255), 1)
top_left = (x, y)
bottom_right = (x + w, y + h)
if w <= 36 and h <= 60:
    x = int((36 - w)/2)
    y = int((60 - h)/2) 
else:
    x1 = w - 36
    y1 = h - 60
    if x1 > y1:
        x = int((w % 3)/2)
        req = (w+x) * 5 / 3
        y = int((req - h)/2)
    else:
        y = int((h % 5)/2)
        req = (y+h) * 3 / 5
        x = int((req - w)/2)
top_left = (top_left[0] - x, top_left[1] - y)
bottom_right = (bottom_right[0] + x, bottom_right[1] + y)        
extracted = face[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
result = cv2.resize(extracted, (36, 60),  interpolation = cv2.INTER_LINEAR)
cv2.imshow('image', face)
cv2.imshow('imag', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('eye.png', 0)
blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
_, contours, _ = cv2.findContours(blurred_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
perimeter = cv2.arcLength(cnt,True)
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
for contour in contours:
    cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
#    cv2.imshow("Frame", img)
cv2.imshow('imag', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('race_track.png')
img = img[100:, 30:-30]
img = cv2.resize(img, (300,300))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#img = cv2.GaussianBlur(img, (11, 11), 0)
#img = cv2.blur(img,(35, 15))
#edges = cv2.Canny(thresh,10,100)
#minLineLength = 100
#maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for i in range(lines.shape[0]):
#    (x1,y1,x2,y2) = lines[i][0]
#    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    
cv2.imshow('imag', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from math import atan2, cos, sin, sqrt, pi
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors = cv2.PCACompute(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    ret,eigenvalues,eigenvectors=cv2.eigen(thresh,True) 
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle

_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c);
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue
    # Draw each contour only for visualisation purposes
    cv2.drawContours(img, contours, i, (0, 0, 255), 2);
    # Find the orientation of each shape
    getOrientation(c, img)
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

img = cv2.imread('notre.jpg', 0)
img = cv2.resize(img, (250, 340))
#img = cv2.GaussianBlur(img, (3, 3), 0)
ret, img = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('notre2.0.png', img)

#img = cv2.imread('RER.jpg', 0)
#img = cv2.resize(img, (264, 131))
#cv2.imshow('output', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('rer.png', img)

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('opencv1.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
k3 = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))
low_filter = cv2.boxFilter(gray, -1, (4,4))
output_low = cv2.filter2D(low_filter, -1, k3)
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(output_low)
plt.title('matrix1')

plt.show()

img, ret, heirarchy = cv2.findContours(output_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#%%    
#function to perform action of states
def action(input_char, replace_with, move, new_state):
    global tapehead, state
    if tape[tapehead] == input_char:
        tape[tapehead] = replace_with
        state = new_state
        if move == 'L':
            tapehead -= 1
        else:
            tapehead += 1
        return True
    return False

string = input("Enter String: ")
length = len(string) + 2
tape = ['B']*length
i = 1
tapehead = 1
for s in string: #loop to place string in tape
    tape[i] = s
    i += 1

state = 0
#assigning characters to variable so that don't have to use characters each time
a, b, X, Z, U, V, R, L, B = 'a', 'b', 'X', 'Z', 'U', 'V', 'R', 'L', 'B' 
oldtapehead = -1
accept = False
while(oldtapehead != tapehead): #if tapehead not moving that means terminate Turing machine
    oldtapehead = tapehead
    print(tape , "with tapehead at index", tapehead, "on state" , state)
    
    if state == 0:
        if action(a, X, R, 1) or action(B, B, R, 10) or action(Z, Z, R, 7) or action(b, U, R, 4):
            pass
        
    elif state == 1:
        if action(a, a, R, 1) or action(b, b, R, 2) or action(B, B, L, 11):
            pass
        
    elif state == 2:
        if action(b, b, R, 2) or action(Z, Z, R, 2) or action(a, Z, L, 3):
            pass
            
    elif state == 3:
        if action(b, b, L, 3) or action(Z, Z, L, 3) or action(a, a, L, 3) or action(X, X, R, 0):
            pass
    
    elif state == 4:
        if action(b, b, R, 4) or action(Z, Z, R, 5) or action(B, B, L, 15):
            pass
        
    elif state == 5:
        if action(Z, Z, R, 5) or action(V, V, R, 5) or action(b, V, L, 6):
            pass
            
    elif state == 6:
        if action(Z, Z, L, 6) or action(V, V, L, 6) or action(b, b, L, 6) or action(U, U, R, 0):
            pass
            
    elif state == 7:
        if action(Z, Z, R, 7) or action(V, V, R, 8):
            pass
            
    elif state == 8:
        if action(V, V, R, 8) or action(B, B, R, 9):
            pass
        
    elif state == 11:
        if action(a, a, L, 11) or action(X, X, R, 12):
            pass
        
    elif state == 12:
        if action(a, Z, R, 13):
            pass
        
    elif state == 13:
        if action(a, X, R, 12) or action(B, B, R, 14):
            pass
            
    elif state == 15:
        if action(b, b, L, 15) or action(U, U, R, 16):
            pass
            
    elif state == 16:
        if action(b, V, R, 17):
            pass
            
    elif state == 17:
        if action(b, U, R, 16) or action(B, B, R, 18):
            pass
            
    else:
        accept = True
        
            
if accept:
    print("String accepted on state = ", state)
else:
    print("String not accepted on state = ", state)
    
#%%
import cv2

img = cv2.imread('Paris (2).jpg', 1)
ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
img = cv2.resize(img, (600, 600))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('final.png', img)

#%%
import cv2
import numpy as np

img = cv2.imread('eg.png')
height, width = img.shape[:2]
img = cv2.resize(img, (int(width/2), int(height/2)), interpolation = cv2.INTER_AREA)
blur = cv2.GaussianBlur(img,(9,9),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
canny = cv2.Canny(mask, 30, 200)
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations = 1)

(_, cnts, heirarchy) = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxx = 0
index = 0
for i in range(0, len(cnts)):
    if max(maxx, cv2.contourArea(cnts[i])) != maxx:
        maxx = cv2.contourArea(cnts[i])
        index = i
#cv2.drawContours(img, cnts, index, (255, 255, 255), 1)

maxx = 0
index2 = -1
for i in range(0, len(cnts)):
    if max(maxx, cv2.contourArea(cnts[i])) != maxx and heirarchy[0, i, 3] == index:
        maxx = cv2.contourArea(cnts[i])
        index2 = i
cv2.drawContours(img, cnts, index2, (255, 255, 255), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
(_, cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        cv2.drawContours(img, c, -1, (0,255,255), -1)

#height, width = img.shape[:2]
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#(_, cnts, heirarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)
#
#maxx = 0
#index = -1
#for i in range(0, len(cnts)):
#    if max(maxx, cv2.contourArea(cnts[i])) != maxx:
#        maxx = cv2.contourArea(cnts[i])
#        index = i
#cv2.drawContours(img, cnts, index, (0, 0, 0), 3)
#
#maxx = 0
#index2 = -1
#for i in range(0, len(cnts)):
#    if max(maxx, cv2.contourArea(cnts[i])) != maxx and heirarchy[0, i, 3] == index:
#        maxx = cv2.contourArea(cnts[i])
#        index2 = i
#cv2.drawContours(img, cnts, index2, (0, 0, 255), 1)

cv2.imshow('image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2

# Grayscale, blur, and threshold
image = cv2.imread('eg.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Fill in potential contours
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
# Remove non rectangular contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,10))
close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Filtered for desired contour
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(approx)
    if len(approx) == 4 and w > h and aspect_ratio > 2.75 and area > 45000:
        cv2.drawContours(image, [c], -1, (36,255,12), -1)
        ROI = original[y:y+h, x:x+w]

cv2.imshow('ROI.png', ROI)
cv2.waitKey(0)

#%%
import cv2

img = cv2.imread('stars.png', 0)
ret, img = cv2.threshold(img,64,255,cv2.THRESH_BINARY)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('star.png', img)
#%%
import cv2
import numpy as np

img = cv2.imread('stars.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
match = cv2.imread('star.png', 0)
h, w = img.shape[:2]
res = np.ones((h, w), np.uint8)*255
ret, thresh = cv2.threshold(gray,64,255,cv2.THRESH_BINARY)
_, cnts, _ = cv2.findContours(match, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key = cv2.contourArea)
area = cv2.contourArea(cnt)
_, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
count = 0
for cnt in cnts:
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt)>=area/2:
        cv2.drawContours(res, cnt, -1, 0, 2)
        count += 1
res = cv2.bitwise_and(img, img, mask = cv2.bitwise_not(res))
cv2.imshow("res", res)
cv2.imshow("star", match)
cv2.imshow("image", img)
cv2.imshow("result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("total count is =", count)

#%%
import cv2
import numpy as np

img = cv2.imread('stars_noise.png')
img = img[10:-10, 10:-10, :]
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,64,255,cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
_, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
res = np.ones((h, w, 3), np.uint8)*255
cv2.drawContours(res, cnts, -1, (0, 255, 0), -1)
cv2.imshow('original', gray)
cv2.imshow('Thresh', thresh)
cv2.imshow("result", res)
cv2.imshow("image", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
cap = cv2.VideoCapture(0)
while(True):
    key = cv2.waitKey(1) & 0xFF
    print(key)
    if key == ord(' '):
        continue
    elif key != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        # Display the resulting frame
        cv2.imshow('frame',frame)
    else:
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

img = cv2.imread('vertical_noise.jpg')
height, width, ch = img.shape
blur = cv2.GaussianBlur(img, (11, 1), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
_, cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(thresh, cnts, -1, 0, 3)
res = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(thresh))
cv2.imwrite('result1.png', res)
blank = np.ones((height, width, ch), np.uint8)*255
inv_res = cv2.bitwise_and(blank, blank, mask=thresh)
cv2.imwrite('result2.png', inv_res)
res = cv2.bitwise_or(res, inv_res)
cv2.imwrite('result3.png', res)
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np
from timeit import default_timer as timer

img = cv2.imread('blob.png', 0)
start = timer()
#_, thresh = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY)
mask = np.ones((3, 3), np.uint8)
res = cv2.filter2D(thresh, -1, mask)
result = np.where(res == np.amax(res))
end = timer()
print(end - start)

#%%
x = 10
i = 0
while i<x:
    i = i+2
    print(i)
    i = i+1

#%%
import cv2

cap = cv2.VideoCapture('guns.gif')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        cv2.imwrite('data\\'+str(i)+'.png', frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

#%%
import cv2

img = cv2.imread('data\\0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
cv2.imshow("img", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

img = cv2.imread('contour.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, cnts, -1, 255, 1)
kernel = np.ones((100, 100), np.uint8)
mask = cv2.dilate(mask, kernel, iterations = 1)
cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cnts, 1, (255, 0, 0), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import pandas as pd

df = []
img = cv2.imread('opencv.png')
img2 = cv2.imread('opencv.png')
histSim = pd.DataFrame(cv2.calcHist(img, [1], None, [256], [0, 256]))
histSim.to_csv('histogram.csv', index=False)

#%%
import cv2

img = cv2.imread('opencv.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY, 11, 2)
thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY, 11, 2)
    
#%%
import os 
name = input("Enter your name ")
path = "/home/rahul/Downloads/opencv-face-recognition/"+name
count = 0
x = os.path.join(path, "frame%d.jpg" % count)

#%%
import cv2
img = cv2.imread('spoon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh2 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
thresh3 = cv2.bitwise_xor(thresh, thresh2)
thresh4 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
cnts, hierarchy = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cnts, -1, (0, 255, 0), -1)
cv2.imshow("img", thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

def gamma_function(channel, gamma):
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    channel = cv2.LUT(channel, table)
    return channel

img = cv2.imread('apply_bokeh.png')
img[:, :, 1] = gamma_function(img[:, :, 1], 1.5)
img[:, :, 2] = gamma_function(img[:, :, 2], 1.5)
# hsv = cv2.
# lower_green = np.array([40,50,50])
# upper_green = np.array([80,255,255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
# res = cv2.bitwise_and(img,img, mask= mask)
img = cv2.GaussianBlur(img,(81,81),0)



cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import math 
  
# This function receives an integer  
# n, and returns the number of 
# digits present in n! 
  
def findDigits(n): 
      
    # factorial exists only for n>=0 
    if (n < 0): 
        return 0; 
  
    # base case 
    if (n <= 1): 
        return 1; 
  
    # else iterate through n and  
    # calculate the value 
    digits = 0; 
    for i in range(2, n + 1): 
        digits += math.log10(i); 
  
    return math.floor(digits) + 1; 

print(findDigits(1500))

#%%
import cv2
import numpy as np

img = cv2.imread('horizontal2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

kernel_hor = np.ones((1, 15), dtype=np.uint8)
erode = cv2.erode(thresh, kernel_hor)
right = erode[:, erode.shape[1]//2:]

kernel = np.ones((3, 3), dtype=np.uint8)
right = cv2.dilate(right, kernel)

cnts, _ = cv2.findContours(right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) > 3:
    print('No need to rotate')
else:
    print('rotate')
    #ADD YOUR ROTATE CODE HERE

cv2.imshow("erode", erode)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

img = cv2.imread('messi.jpg')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_color = (255, 255, 255)
thick = 1
text = "A very long text here blaaah blaaah blaaah blaaah blaaah blaaah . . . . . "
font_size = 0.9
(text_width, text_height) = cv2.getTextSize(text, font, font_size, thick)[0]
text_height += 15

mask = np.zeros((text_height, text_width), dtype=np.uint8)
mask = cv2.putText(mask,text,(0,15),font,font_size,font_color,thick,cv2.LINE_AA)
mask = cv2.resize(mask, (img.shape[1], text_height))

mask = cv2.merge((mask, mask, mask))

img[-text_height:, :, :] = cv2.bitwise_or(img[-text_height:, :, :], mask)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense

input_layer = Input(shape=(224,224,3)) #Image resolution is 224x224 pixels
x = Conv2D(128, (7, 7), padding='same', activation='relu', strides=(2, 2))(input_layer)

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top = False, input_shape=x.shape[0])
for layer in inception_model.layers:
    layer.trainable = False

x = inception_model (x) #Error in this line
x = GlobalAveragePooling2D()(x)

predictions = Dense(11, activation='softmax')(x) #I have 11 classes of image to classify

model = tf.keras.models.Model(inputs = input_layer, outputs=predictions)

#%%
import cv2
import numpy as np

img = cv2.imread('shadow.jpg')
original = cv2.imread('bright.jpg')
height, width = img.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(width, 150)
kernel_y = cv2.getGaussianKernel(height, 150)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)

test = img.copy()
for i in range(3):
    test[:, :, i] = test[:, :, i] / mask    
    
hsv = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
hsv = np.array(hsv, dtype = np.float64)
hsv[:,:,1] = hsv[:,:,1]*1.3 ## scale pixel values up or down for channel 1(Lightness)
hsv[:,:,1][hsv[:,:,1]>255]  = 255
hsv[:,:,2] = hsv[:,:,2]*1.3 ## scale pixel values up or down for channel 1(Lightness)
hsv[:,:,2][hsv[:,:,2]>255]  = 255
hsv = np.array(hsv, dtype = np.uint8)
test = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


cv2.imshow('Original_bright', original)
cv2.imshow('Original_dark', img)
cv2.imshow('Result', test)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

img = cv2.imread('2.jpg')
res = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([40,50,50])
upper_green = np.array([80,255,255])
r2, g2, b2 = 255, 0, 0

mask = cv2.inRange(hsv, lower_green, upper_green)

mask = mask/255
mask = mask.astype(np.bool)
res[:,:,:3][mask] = [b2, g2, r2] # opencv uses BGR

cv2.imshow('image', img)
cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2

cap = cv2.VideoCapture(0)
i = 0
for i in range(100):
    ret, frame = cap.read()
    cv2.imwrite('images/'+str(i)+'.jpg', frame)
cap.release()

#%%
