import pytesseract
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    	
    (thresh, black) = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
    kernel = np.ones((8, 8), np.uint8)
    eroded = cv.erode(black, kernel, iterations=1)
    seed=~eroded
    cv.imshow('SEED', seed)
    text = pytesseract.image_to_string(seed, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    print('>'+text)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    # cv.imshow('mask',mask)
    # cv.imshow('gray',gray)
    # cv.imshow('black',black)
    # cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()