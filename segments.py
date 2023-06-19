import cv2 as cv
import numpy as np
import pytesseract as pyt
from Util import crop


def readseg(seg):
    # Convert to gray
    gray = cv.cvtColor(seg, cv.COLOR_BGR2GRAY)
    # Convert to BW
    (th, bw) = cv.threshold(gray, 100, 255, cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv.erode(bw, kernel, iterations=2)
    cv.imshow('Eroded', eroded)

    inverted = ~eroded
    contours, hierarchy = cv.findContours(
        eroded,  cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)
    with_contours = cv.drawContours(seg, contours, 1,(0,255,255),1)
    cv.imshow('Detected contours', with_contours)
    

    # # lower bound and upper bound for Green color
    # lower_bound = np.array([50, 20, 20])
    # upper_bound = np.array([100, 255, 255])

    # # find the colors within the boundaries
    # mask = cv.inRange(bw, lower_bound, upper_bound)

    # #define kernel size
    # kernel = np.ones((7,7),np.uint8)

    # # Remove unnecessary noise from mask
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # # Segment only the detected region
    # segmented_img = cv.bitwise_and(bw, bw, mask=mask)

    # # Find contours from the mask

    # contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.HAIN_APPROX_SIMPLE)
    # output = cv.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    # Showing the output
    # cv.imshow("Output", output)

    # contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # max_area = 0
    # max_contour = None
    # for contour in contours:
    #     area = cv.contourArea(contour)
    #     if area > max_area:
    #         max_area = area
    #         max_contour = contour
    # x, y, w, h = cv.boundingRect(max_contour)
    # contour_image = bw.copy()
    # cv.rectangle(contour_image, (x,y),(x+h,y+h), (0, 255, 0), 5)
    # cv.imshow('Contour Image', contour_image)

    # th=0.2
    # tv=2*th
    # h, w = orig.shape[:2];
    # flags = [];
    # segments = [];
    # h1 = [[0, 1.0],[0, th]];       # 0
    # h2 = [[0, 1.0],[0.5-th/2, 0.5+th/2]];   # 1
    # h3 = [[0, 1.0],[1.0-th, 1.0]];     # 2
    # vl1 = [[0, tv],[0, 0.5]];      # 3 # upper-left
    # vl2 = [[0, tv],[0.5, 1.0]];    # 4
    # vr1 = [[1.0-tv, 1.0],[0, 0.5]];    # 5 # upper-right
    # vr2 = [[1.0-tv, 1.0], [0.5, 1.0]]; # 6
    # segments.append(h1);
    # segments.append(h2);
    # segments.append(h3);
    # segments.append(vl1);
    # segments.append(vl2);
    # segments.append(vr1);
    # segments.append(vr2);

    # for a,seg in enumerate(segments):
    #     xl, xh = seg[0];
    #     yl, yh = seg[1];
    #     # convert to pix coords
    #     xl = int(xl * w);
    #     xh = int(xh * w);
    #     yl = int(yl * h);
    #     yh = int(yh * h);
    #     sw = xh - xl;
    #     sh = yh - yl;

    #     # check
    #     count = np.count_nonzero(orig[yl:yh, xl:xh] < 200);
    #     if count / (sh * sw) > 0.1: # 0.5 is a sensitivity measure
    #         flags.append(a);
    #     overlay=orig.copy()
    #     cv.rectangle(overlay, (xl, yl), (xh, yh), (0, 0,255), -1)
    #     alpha = 0.5
    #     orig = cv.addWeighted(overlay, alpha, orig, 1 - alpha, 0)
    #     cv.imshow('segm',orig)
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

    return 0
