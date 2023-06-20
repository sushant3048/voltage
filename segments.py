import cv2 as cv
import numpy as np
# import pytesseract as pyt
from Util import crop


def readseg(inp,id):
    # try: 
        h,w,t=inp.shape
        aspect=w/h
        h=500
        w=int(h*aspect)
        slice=cv.resize(inp,(w,h))
        shapes=np.zeros_like(slice,np.uint8)


        # disp=slice.copy()
        gray = cv.cvtColor(slice, cv.COLOR_BGR2GRAY)
        # Convert to BW
        (th, bw) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

        kernel = np.ones((4, 4), np.uint8)
        eroded = cv.erode(bw, kernel, iterations=1)
        # cv.imshow('Eroded', eroded)
        inverted = ~eroded
        disp=cv.cvtColor(inverted, cv.COLOR_GRAY2BGR)

        x=0
        y=0 
        h,w,d=slice.shape

        
        # contours, hierarchy = cv.findContours(
        #     inverted,  cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
        # slice = cv.drawContours(slice, contours, -1,(0,255,255),1)
        
        # max_contour=max(contours, key=cv.contourArea)
        # x,y,w,h=cv.boundingRect(max_contour)
        # print('rec',x,y,w,h)
        # special case of 1
        # rec 8 12 105 207
        # rec 82 26 36 186
        # if w/h<0.5:
        #     x=int(x-2*w)
        #     w=3*w


        # cv.rectangle(slice,(x,y),(x+w,y+h),(0,0,255),1)
        # cv.imshow('Detected contours', slice)

        # eroded=crop(eroded,[(x,y),[x+w,y+h]])
        # disp=crop(disp,[(x,y),[x+w,y+h]])


        # Digit detection ////////////////////////
        th=0.30
        tv=0.20
        # h, w = orig.shape[:2];
        flags = [];
        segments = [];
        # define rectangels in x,y w,h format
        h1 = [th,0,1-2*th,tv];       # 0
        h2 = [th,0.5-tv/2,1-2*th,tv];   # 1
        h3 = [th,1-tv,1-2*th,tv];     # 2
        vl1 = [0,tv,th,0.5-1.5*tv];      # 3 # upper-left
        vl2 = [0,0.5+tv/2,th,0.5-1.5*tv];    # 4
        vr1 = [1-th,tv,th,0.5-1.5*tv];    # 5 # upper-right
        vr2 = [1-th,0.5+tv/2,th,0.5-1.5*tv];  # 6
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
            part=crop(eroded, [(x,y),(x+wd,y+ht)])
            count = np.count_nonzero(part==0);
            area=wd*ht
            # print('count, area, coverage',count,area, round(count/area*100))
            if count / area > 0.1: # 0.5 is a sensitivity measure
                flags.append(a);
            cv.rectangle(shapes, (x,y), (x+wd, y+ht), (0,255,255), cv.FILLED)
            alpha=0.9
            mask=shapes.astype(bool)
            temp=disp.copy()
            disp[mask]=cv.addWeighted(temp,alpha,shapes,1-alpha,0)[mask]
        cv.imshow(id,disp)

        if flags == [0,2,3,4,5,6]:
            return '0';
        if flags == [5,6]:
            return '1';
        if flags == [0,1,2,4,5]:
            return '2';
        if flags == [0,1,2,5,6]:
            return '3';
        if flags == [1,3,5,6]:
            return '4';
        if flags == [0,1,2,3,6]:
            return '5';
        if flags == [0,1,2,3,4,6]:
            return '6';
        if flags == [0,5,6]:
            return '7';
        if flags == [0,1,2,3,4,5,6]:
            return '8';
        if flags == [0,1,2,3,5,6]:
                return '0';
        return None
        print(flags)
        
    # except Exception as e: # work on python 3.x
    #     print(str(e))
        return None;
       

