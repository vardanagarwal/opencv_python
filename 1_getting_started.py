# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:33:20 2019

@author: hp
"""

import cv2

img = cv2.imread('image_name.png') # to read image

cv2.imshow("image", img) # to display image
cv2.waitKey(0) #Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event.
#If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke.
cv2.destroyAllWindows() # cv2.destroyAllWindows() simply destroys all the windows we created

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to convert to grayscale

cv2.imwrite('img_gray.png', gray) # to write image

cap = cv2.VideoCapture(0) # to take input from webcam
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == 27:
        break

#%%
#VIDEO

import cv2

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# read a video and display it after rotating it by 180 degrees
cap = cv2.VideoCapture('video.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()