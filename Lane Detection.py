from __future__ import print_function
import cv2 as cv
import numpy as np

# Define HSV ranges for blue and yellow
lower_blue = np.array([60, 35, 140])
upper_blue = np.array([180, 255, 255])

lowerYellow = np.array([10, 90, 90])
upperYellow = np.array([40, 255, 255])

# Open video capture
capture = cv.VideoCapture(r"C:\Users\james\OneDrive\Documents\GitHub\ComputerVision\IMG_0625.MP4")



while True:
    isTrue, frame = capture.read()

    # Convert frame to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Apply yellow mask
    yellowMask = cv.inRange(hsv, lowerYellow, upperYellow)
    yellowResult = cv.bitwise_and(frame, frame, mask=yellowMask)

    # Apply blue mask
    blueMask = cv.inRange(hsv, lower_blue, upper_blue)
    blueResult = cv.bitwise_and(frame, frame, mask=blueMask)
    
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
