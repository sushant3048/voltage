from Util import capture, pyread1, pyread2
import cv2
import numpy as np

while(1):
    img=capture()
    
    # lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
    l,a,b = cv2.split(lab);

    # show
    cv2.imshow("orig", img);

    # threshold params
    low = 105;
    high = 255;
    iters = 3;

    # make copy
    copy = b.copy();

    # threshold
    thresh = cv2.inRange(copy, low, high);
    kernel = np.ones((5,5), np.uint8);
    # dilate
    for a in range(iters):
        thresh = cv2.dilate(thresh, kernel);

    # erode
    for a in range(iters):
        thresh = cv2.erode(thresh, kernel);

    # show image
    cv2.imshow("thresh", thresh);
    # cv2.imwrite("threshold.jpg", thresh);

    # start processing
    # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    k = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    contours=k[0]

    # draw
    for contour in contours:
        cv2.drawContours(img, [contour], 0, (0,255,0), 3);
    
    cv2.imshow("contour", img);

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
