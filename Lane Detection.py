from __future__ import print_function
import cv2 as cv
import numpy as np

# Define HSV ranges for blue and yellow
lower_blue = np.array([60, 35, 140])
upper_blue = np.array([180, 255, 255])

lowerYellow = np.array([30, 40, 40])
upperYellow = np.array([60, 255, 255])

# Open video capture
capture = cv.VideoCapture(r"C:\Users\james680384\Documents\GitHub\ComputerVision\IMG_0625.mov")


while True:
    isTrue, frame = capture.read()

    resizedFrame = cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)

    # Convert frame to HSV
    hsv = cv.cvtColor(resizedFrame, cv.COLOR_BGR2HSV)
    
    # Apply yellow mask
    yellowMask = cv.inRange(hsv, lowerYellow, upperYellow)
    yellowResult = cv.bitwise_and(resizedFrame, resizedFrame, mask=yellowMask)

    # Apply blue mask
    blueMask = cv.inRange(hsv, lower_blue, upper_blue)
    blueResult = cv.bitwise_and(resizedFrame, resizedFrame, mask=blueMask)
    
    finalResult = yellowResult + blueResult

    # Display results
    cv.imshow('Blue Result', blueResult)
    cv.imshow('Yellow Result', yellowResult)
    cv.imshow('Final Result', finalResult)

    cv.namedWindow('Final Result', cv.WINDOW_AUTOSIZE)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release capture and destroy windows
capture.release()
cv.destroyAllWindows()
