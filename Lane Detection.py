from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

lower_blue = np.array([60, 35, 140]) 
upper_blue = np.array([180, 255, 255]) 

lowerYellow = np.array([20, 100, 100])
upperYellow = np.array([30, 255, 255])


capture = cv.VideoCapture(r"C:\Users\james\OneDrive\Documents\GitHub\ComputerVision\IMG_0625.MP4")

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(hsv, (1,1))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = frame * (mask[:, :, None].astype(frame.dtype))
    cv.imshow(window_name, dst)

cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

while True:
    isTrue, frame = capture.read()
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    blueMask = cv.inRange(hsv, lower_blue, upper_blue) 

    blueResult = cv.bitwise_and(frame, frame, mask = blueMask) 
    
    yellowMask = cv.inRange(hsv, lowerYellow, upperYellow) 

    cv.imshow('Yellow Mask', yellowMask)

    yellowResult = cv.bitwise_and(frame, frame, mask = blueMask) 

    cv.imshow('Result', yellowResult)
    cv.imshow('Result', blueResult)

    low_threshold = cv.getTrackbarPos(title_trackbar, window_name)
    CannyThreshold(low_threshold)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
