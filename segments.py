import cv2 as cv
import numpy as np
import pytesseract as pyt
from Util import crop


def readseg(slice):
    # Convert to gray
    disp=slice.copy()
    gray = cv.cvtColor(slice, cv.COLOR_BGR2GRAY)
    # Convert to BW
    (th, bw) = cv.threshold(gray, 100, 255, cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv.erode(bw, kernel, iterations=2)
    # cv.imshow('Eroded', eroded)

    inverted = ~eroded
    contours, hierarchy = cv.findContours(
        eroded,  cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
    with_contours = cv.drawContours(slice, contours, 1,(0,255,255),1)
    # cv.imshow('Detected contours', with_contours)
    max_contour=max(contours, key=cv.contourArea)
    x,y,w,h=cv.boundingRect(max_contour)
    cropped=crop(eroded,[(x,y),[x+w,y+h]])
    disp=crop(disp,[(x,y),[x+w,y+h]])
    # disp = cv.cvtColor(disp, cv.COLOR_GRAY2BGR)
    

    # Digit detection ////////////////////////
    th=0.1
    tv=0.1
    # h, w = orig.shape[:2];
    flags = [];
    segments = [];
    h1 = [[0, 1.0],[0, th]];       # 0
    h2 = [[0, 1.0],[0.5-th/2, 0.5+th/2]];   # 1
    h3 = [[0, 1.0],[1.0-th, 1.0]];     # 2
    vl1 = [[0, tv],[0, 0.5]];      # 3 # upper-left
    vl2 = [[0, tv],[0.5, 1.0]];    # 4
    vr1 = [[1.0-tv, 1.0],[0, 0.5]];    # 5 # upper-right
    vr2 = [[1.0-tv, 1.0], [0.5, 1.0]]; # 6
    segments.append(h1);
    segments.append(h2);
    segments.append(h3);
    segments.append(vl1);
    segments.append(vl2);
    segments.append(vr1);
    segments.append(vr2);

    for a,seg in enumerate(segments):
        xl, xh = seg[0];
        yl, yh = seg[1];

        # convert to pix coords
        xl = int(xl * w);
        xh = int(xh * w);
        yl = int(yl * h);
        yh = int(yh * h);
        sw = xh - xl;
        sh = yh - yl;

        # check
        count = np.count_nonzero(eroded[yl:yh, xl:xh] < 200);
        area=sh*sw
        print(count,area, round(count/area*100))
        if count / area > 0.1: # 0.5 is a sensitivity measure
            flags.append(a);
        cv.rectangle(disp, (xl, yl), (xh, yh), (0, 0,255), -1)
    cv.imshow('letter',disp)
    # # print(flags)
    # if flags == [0,2,3,4,5,6]:
    #     return 0;
    # if flags == [5,6]:
    #     return 1;
    # if flags == [0,1,2,4,5]:
    #     return 2;
    # if flags == [0,1,2,5,6]:
    #     return 3;
    # if flags == [1,3,5,6]:
    #     return 4;
    # if flags == [0,1,2,3,6]:
    #     return 5;
    # if flags == [0,1,2,3,4,6]:
    #     return 6;
    # if flags == [0,5,6]:
    #     return 7;
    # if flags == [0,1,2,3,4,5,6]:
    #     return 8;
    # if flags == [0,1,2,3,5,6]:
    #         return 9;
    # return -1;

    # return 0
