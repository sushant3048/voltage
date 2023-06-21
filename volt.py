import cv2 as cv
import numpy as np
from util import crop
from segments import readseg

cap = cv.VideoCapture(0)
prev = "000"
while True:
    _, inp = cap.read()
    hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)

    # Define the lower and upper blue thresholds
    lower_blue = np.array([95, 0, 0])
    upper_blue = np.array([105, 255, 255])

    # Create a mask for blue pixels
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if len(contours) > 0:
        max_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(max_contour)

        # rectangle the detected area
        cv.rectangle(inp, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # tightening the rectangle inward symmetrically
        x = int(x+0.1*h)
        y = int(y+0.05*h)
        w = int(w-0.2*h)
        h = int(h-0.1*h)

        # width further reduced to correct the right offset due to presence of 'v'
        # Centering is important for correct partitioning of the digits
        w = int(w-0.06*h)

        # rectangle the area of interest
        cv.rectangle(inp, (x, y), (x+w, y+h), (0, 255, 255), 1)

        # Display the input feed with rectangels.
        cv.imshow('input', inp)
        #--------
        cv.imshow('hsv',hsv)
        cv.imshow('mask',mask)

        myimg=mask.copy()
        print(inp.shape[:2])
       


        #--------


        # Crop the area of interest from ..
        frame=crop(mask,[x,y,w,h])
      

        # big=cv.resize(a,(1000,int(1000*h/w)))
        # gray = cv.cvtColor(big, cv.COLOR_BGR2GRAY)
        # (th, bw) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

        # kernel = np.ones((10, 10), np.uint8)
        # dilated = cv.dilate(bw, kernel, iterations=5)
        # cv.imshow('dil',dilated)
        # inverted = ~dilated
        # eroded = cv.dilate(inverted, kernel, iterations=6)

        # cv.imshow('ero',eroded)

        # cropped = crop(inp, [x, y, w, h])

        # # setting refrences in respect of cropped image
        # # x=0
        # # y=0
        # # slicing into 3 parts for each digit.
        # wd = int(w/3)
        # x1 = x+wd
        # x2 = x1+wd

        # # cords of each digit
        # r1 = [(x, y), (x1, y+h)]
        # r2 = [(x1, y), (x2, y+h)]
        # r3 = [(x2, y), (x+w, y+h)]

        # disp = cropped.copy()
        # cv.rectangle(inp, *r1, (0, 255, 255), 2)
        # cv.rectangle(inp, *r2, (0, 255, 255), 2)
        # cv.rectangle(inp, *r3, (0, 255, 255), 2)

        # cv.imshow('inp', inp)

        # # # idenfity the digit
        # seg1 = crop(inp, [x, y, wd, h])
        # seg2 = crop(inp, [x1, y, wd, h])
        # seg3 = crop(inp, [x2, y, wd, h])
        # d1 = readseg(seg1, 'first')
        # d2 = readseg(seg2, 'second')
        # d3 = readseg(seg3, 'third')
        # try:
        #     val = ""+d1+d2+d3
        #     prev = val
        # except:
        #     val = prev

        # print(val)

        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
