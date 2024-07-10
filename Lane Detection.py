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
    # Merge the image with the lines onto the original.

    return img

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

    blueLines = cv.HoughLinesP(
    blueMask,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
    )
    print(blueLines)

    yellowLines = cv.HoughLinesP(
    blueMask,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25)

    print(yellowLines)

    blueLineImage = draw_lines(frame, blueLines)
    cv.imshow('Blue Lines', blueLineImage)

    yellowLineImage = draw_lines(frame, yellowLines)
    cv.imshow('Yellow Lines', yellowLineImage)

    finalLines = blueLineImage + yellowLineImage

    final = cv.addWeighted(frame, 0.8, finalLines, 1.0, 0.0)

    cv.imshow('Final', final)    

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release capture and destroy windows
capture.release()
cv.destroyAllWindows()
