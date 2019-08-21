# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:10:50 2019

@author: hp
"""

import automatic_soduku_solver
import cv2
import pytesseract

img = cv2.imread('sudoku_processed.png')
frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, cnts, _ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, cnts, -1, (255,255,255), 3)
ret, thresh = cv2.threshold(frame,100,255,cv2.THRESH_BINARY)
width, height = thresh.shape[::-1]
x = int(width/9)
y = int(height/9)
array = [[0 for i in range(9)] for j in range(9)]
flag = [[0 for i in range(9)] for j in range(9)]
for i in range(9):
    for j in range(9):
        im = thresh[i*x:(i+1)*x, j*y:(j+1)*y]
        if im.shape[0]*im.shape[1] - sum(map(sum, im[:, :]))/255 == 0:
            array[i][j] = 0
        else:
            array[i][j] = int(pytesseract.image_to_string(im, lang="eng", 
                 config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'))
            flag[i][j] = 1

def print_grid(arr): 
    for i in range(9): 
        for j in range(9): 
            print (arr[i][j], end = ' ') 
        print () 

def find_empty_location(arr,l): 
    for row in range(9): 
        for col in range(9): 
            if(arr[row][col]==0): 
                l[0]=row 
                l[1]=col 
                return True
    return False

def used_in_row(arr,row,num): 
    for i in range(9): 
        if(arr[row][i] == num): 
            return True
    return False
   
def used_in_col(arr,col,num): 
    for i in range(9): 
        if(arr[i][col] == num): 
            return True
    return False

def used_in_box(arr,row,col,num): 
    for i in range(3): 
        for j in range(3): 
            if(arr[i+row][j+col] == num): 
                return True
    return False
  
def check_location_is_safe(arr,row,col,num): 
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num) 

def solve_sudoku(arr):       
    l=[0,0]    
    if(not find_empty_location(arr,l)): 
        return True
    row=l[0] 
    col=l[1] 
    for num in range(1,10): 
        if(check_location_is_safe(arr,row,col,num)): 
            arr[row][col]=num 
            if(solve_sudoku(arr)): 
                return True
            arr[row][col] = 0
    return False 

if(solve_sudoku(array)): 
        print_grid(array) 
else: 
    print ("No solution exists")
    
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(9):
    for j in range(9):
        if flag[i][j] == 0:
            cv2.putText(img,str(array[i][j]),(j*y + 10, (i+1)*x - 10), font, 1,(0,255,0),2,cv2.LINE_AA)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()