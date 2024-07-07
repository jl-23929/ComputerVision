import cv2 as cv

capture = cv.VideoCapture(0)


while True:
    isTrue, frame = capture.read()

    cv.imshow('Video', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    blur = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT)
    cv.imshow('Blur', blur)

    canny = cv.Canny(blur, 125, 175)
    cv.imshow('Canny', canny)

    dilated = cv.dilate(canny, (3,3), iterations=5)
    cv.imshow('Dilated', dilated)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break   