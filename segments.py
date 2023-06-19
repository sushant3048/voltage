import cv2 as cv
import numpy as np
import pytesseract as pyt
from Util import crop


def readseg(slice):
    # canvas = np.zeros([1024,512,3],dtype=np.uint8)
    # canvas.fill(255)

    print('slice',slice.shape)
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
        inverted,  cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
    with_contours = cv.drawContours(slice, contours, -1,(0,255,255),1)
    
    max_contour=max(contours, key=cv.contourArea)
    x,y,w,h=cv.boundingRect(max_contour)
    cv.rectangle(with_contours,(x,y),(x+w,y+h),(0,0,255),1)
    cv.imshow('Detected contours', with_contours)

    cropped=crop(eroded,[(x,y),[x+w,y+h]])
    disp=crop(disp,[(x,y),[x+w,y+h]])


    # Digit detection ////////////////////////
    th=0.25
    tv=0.10
    # h, w = orig.shape[:2];
    flags = [];
    segments = [];
    # define rectangels in x,y w,h format
    h1 = [0,0,1,tv];       # 0
    h2 = [0,0.5-tv/2,1,tv];   # 1
    h3 = [0,1-tv,1,tv];     # 2
    vl1 = [0,0,th,0.5];      # 3 # upper-left
    vl2 = [0,0.5,th,0.5];    # 4
    vr1 = [1-th,0,th,0.5];    # 5 # upper-right
    vr2 = [1-th,0.5,th,0.5];  # 6
    segments.append(h1);
    segments.append(h2);
    segments.append(h3);
    segments.append(vl1);
    segments.append(vl2);
    segments.append(vr1);
    segments.append(vr2);

    for a,seg in enumerate(segments):
        x,y,wd,ht=seg
        x=int(x*w)
        y=int(y*h)
        wd=int(wd*w)
        ht=int(ht*h)

        # check
        part=crop(cropped, [(x,y),(x+wd,y+ht)])
        print('--------------')
        print(a)
        print('---------------')
        print(part)
        print('part sixe',part.shape)
        count = np.count_nonzero(part==0);
        area=wd*ht
        print('count, area, coverage',count,area, round(count/area*100))
        if count / area > 0.1: # 0.5 is a sensitivity measure
            flags.append(a);
        cv.rectangle(cropped, (x,y), (x+wd, y+ht), (0, 0,255), 1)
    print('disp',disp.shape)
    cv.imshow('letter',eroded)
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
