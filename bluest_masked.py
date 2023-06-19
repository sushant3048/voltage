import cv2 as cv
import numpy as np
import pytesseract as pyt
import matplotlib.pyplot as plt


def segread(obj):
    gray = cv.cvtColor(obj, cv.COLOR_BGR2GRAY)
    cv.imshow('gray',gray)
    (th, bw) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    x, y, w, h = cv.boundingRect(max_contour)
    contour_image = bw.copy()
    cv.rectangle(contour_image, (x,y),(x+h,y+h), (0, 255, 0), 5)
    cv.imshow('Contour Image', contour_image)

    th=0.2
    tv=2*th
    h, w = obj.shape[:2];
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
        count = np.count_nonzero(obj[yl:yh, xl:xh] < 200);
        if count / (sh * sw) > 0.1: # 0.5 is a sensitivity measure
            flags.append(a);
        overlay=obj.copy()
        cv.rectangle(overlay, (xl, yl), (xh, yh), (0, 0,255), -1)  
        alpha = 0.5
        obj = cv.addWeighted(overlay, alpha, obj, 1 - alpha, 0)
        cv.imshow('segm',obj)
    # print(flags)
    if flags == [0,2,3,4,5,6]:
        return 0;
    if flags == [5,6]:
        return 1;
    if flags == [0,1,2,4,5]:
        return 2;
    if flags == [0,1,2,5,6]:
        return 3;
    if flags == [1,3,5,6]:
        return 4;
    if flags == [0,1,2,3,6]:
        return 5;
    if flags == [0,1,2,3,4,6]:
        return 6;
    if flags == [0,5,6]:
        return 7;
    if flags == [0,1,2,3,4,5,6]:
        return 8;
    if flags == [0,1,2,3,5,6]:
            return 9;
    
    return -1;




def find_bluest_object(frame):
    # Convert the frame to the HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the lower and upper blue thresholds
    lower_blue = np.array([95, 0, 0])
    upper_blue = np.array([105, 255, 255])

    # Create a mask for blue pixels
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.imshow('mask', mask)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Return the bounding rectangle of the bluest object
    if max_contour is not None:
        x, y, w, h = cv.boundingRect(max_contour)
        return (x, y, w, h)
    else:
        return None


def crop(img, bounds):
    (x1, y1), (x2, y2) = bounds
    cropped = img[y1:y2, x1:x2]
    return cropped








# Open the webcam
cap = cv.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    dup = frame.copy()

    # Find the bluest object in the frame
    rectangle = find_bluest_object(frame)

    if rectangle is not None:
        # Draw a rectangle around the bluest object
        x, y, w, h = rectangle

        # tightening the rectangle with experimental values
        x = int(x+w*0.07)
        w = int(w*0.85)
        y = int(y+h*0.05)
        h = int(h*0.90)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        (th, frame) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

        black_mask = np.zeros(frame.shape,np.uint8)
        white_mask=cv.bitwise_not(black_mask)
        white_mask[y:y+h,x:x+w] = frame[y:y+h,x:x+w]
        frame=white_mask
        
      
        # t = pyt.image_to_string(frame, config='--psm 13')
        # t = pyt.image_to_string(frame, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        # print('>>'+t)


        
        # slicing into 3 parts for each digit.
        wd = int(w/3)
        x1 = x+wd
        x2 = x1+wd

        # cords of each digit
        r1 = [(x, y), (x1, y+h)]
        r2 = [(x1, y), (x2, y+h)]
        r3 = [(x2, y), (x+w, y+h)]

        # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.rectangle(frame, *r1, (0, 255, 255), 2)
        cv.rectangle(frame, *r2, (0, 255, 255), 2)
        cv.rectangle(frame, *r3, (0, 255, 255), 2)
        cv.imshow('masked',frame)

        # idenfity the digit
        i1 = crop(dup, r1)
        i2 = crop(dup, r2)
        i3 = crop(dup, r3)

        # d1=segread(i1)
        d2=segread(i2)
        # d3=segread(i3)
        # print(d2)

        # print('>'+str(d1)+str(d2)+str(d3))

        
        # i1 = cv.cvtColor(i1, cv.COLOR_BGR2RGB)
#         t = pyt.image_to_string(i1, lang='eng', config='--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')
#         print('>'+t)
# # ///////////////////////////////////////////
        # kernel = np.ones((5,5), np.uint8);

        # # dilate
        # i1 = cv.dilate(i1, kernel, iterations=1);

        # erode
        # for a in range(iters):
        #     thresh = cv.erode(thresh, kernel);
         

    # Display the frame
    cv.imshow('Webcam', dup)

    # Check if 'q' key is pressed to exit
    if cv.waitKey(1) == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv.destroyAllWindows()
