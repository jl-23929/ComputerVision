from __future__ import print_function
import cv2 as cv
import numpy as np
import math

# Define HSV ranges for blue and yellow
lower_blue = np.array([60, 35, 140])
upper_blue = np.array([180, 255, 255])

lowerYellow = np.array([30, 40, 40])
upperYellow = np.array([60, 255, 255])

# Open video capture
capture = cv.VideoCapture(r"C:\Users\james\OneDrive\Documents\GitHub\ComputerVision\IMG_0625.MP4")

def draw_lines(img, lines, color=[255, 255, 255], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:

        for x1, y1, x2, y2 in line:
            cv.line(line_img, (x1, y1), (x2, y2), color, thickness)

    return line_img

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

    blueLines = cv.Canny(blueMask, 50, 150, apertureSize=3)

    yellowLines = cv.Canny(yellowMask, 50, 150, apertureSize=3)

    finalLineImage = blueLines + yellowLines
    
    lines = cv.HoughLinesP(finalLineImage, 5, np.pi/180, 50, maxLineGap=50)

    line_image = draw_lines(resizedFrame, lines)

    cv.imshow('Line Image', line_image)

    print(lines)


    raycastingPoint = (320, 480)

    maxDistance = 0

    longestEndpoint = None

    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line[0]

            intersectionY = y2

            intersectionX = x2

            distance = math.sqrt((intersectionX - raycastingPoint[0]) ** 2 + (intersectionY - raycastingPoint[1]) ** 2)

            gradient = (y2 - y1) / (x2 - x1)

            intercept = y1 - gradient * x1

            

            if distance > maxDistance:
                maxDistance = distance
                longestEndpoint = (intersectionX, intersectionY)
    
    if longestEndpoint is not None:
        cv.line(line_image, raycastingPoint, longestEndpoint, (0, 255, 0), 3)

    cv.imshow('Final', line_image)    

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release capture and destroy windows
capture.release()
cv.destroyAllWindows()
