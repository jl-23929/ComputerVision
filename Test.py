import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture(0)

blank = np.zeros((500,500,3), dtype='uint8')
blank[200:300, 300:400] = 0,255,0
cv.imshow('Green', blank)

cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.imshow('Rectangle', blank)

cv.circle(blank, (300,300), 40, (0,0,255), thickness=2)
cv.putText(blank, 'Hello World', (255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness=2)
cv.imshow('Circle', blank)

while True:
    isTrue, frame = capture.read()

    resizedFrame = rescaleFrame(frame, scale=0.2)
    cv.imshow('Video', frame)
    cv.imshow('Resized Video', resizedFrame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()