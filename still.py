import cv2 as cv
import numpy as np
# import pytesseract as pyt
from Util import *
from segments import readseg
# import matplotlib.pyplot as plt

inp=cv.imread('media/211.jpg')
hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)

# Define the lower and upper blue thresholds
lower_blue = np.array([95, 0, 0])
upper_blue = np.array([105, 255, 255])

# Create a mask for blue pixels
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Find contours in the mask
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
if len(contours) > 0:
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)
    

    # tightening the rectangle with experimental values
    x = int(x+w*0.07)
    w = int(w*0.85)
    y = int(y+h*0.05)
    h = int(h*0.90)

    cropped = crop(inp,[(x,y),(x+w,y+h)])

    # setting refrences in respect of cropped image
    # x=0
    # y=0
    # slicing into 3 parts for each digit.
    wd = int(w/3)
    x1 = x+wd
    x2 = x1+wd


    # cords of each digit
    r1 = [(x, y), (x1, y+h)]
    r2 = [(x1, y), (x2, y+h)]
    r3 = [(x2, y), (x+w, y+h)]

    disp=cropped.copy()
    cv.rectangle(inp, *r1, (0, 255, 255), 2)
    cv.rectangle(inp, *r2, (0, 255, 255), 2)
    cv.rectangle(inp, *r3, (0, 255, 255), 2)

    cv.imshow('inp',inp)


    # # idenfity the digit
    seg1 = crop(inp, r1)
    seg2 = crop(inp, r2)
    seg3 = crop(inp, r3)
    d1=readseg(seg1)
    d2=readseg(seg2)
    d3=readseg(seg3)
    print('>'+d1+d2+d3)

    cv.waitKey(0)
    cv.destroyAllWindows()


