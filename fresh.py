import cv2 as cv
import numpy as np
import pytesseract as pyt
from Util import *
from segments import readseg
# import matplotlib.pyplot as plt

inp=cv.imread('media/210.jpg')
hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)

# Define the lower and upper blue thresholds
lower_blue = np.array([95, 0, 0])
upper_blue = np.array([105, 255, 255])

# Create a mask for blue pixels
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Find contours in the mask
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
max_area = 0
max_contour = None
for contour in contours:
    area = cv.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

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
cv.rectangle(inp, *r1, (0, 255, 0), 2)
cv.rectangle(inp, *r2, (0, 255, 0), 2)
cv.rectangle(inp, *r3, (0, 255, 0), 2)

cv.imshow('inp',inp)

d1=readseg(inp,r1)
# # idenfity the digit
# i1 = crop(dup, r1)
# i2 = crop(dup, r2)
# i3 = crop(dup, r3)

cv.waitKey(0)
cv.destroyAllWindows()

# gray = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)
# (th, frame) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
# black_mask = np.zeros(frame.shape,np.uint8)
# white_mask=cv.bitwise_not(black_mask)
# white_mask[y:y+h,x:x+w] = frame[y:y+h,x:x+w]
# frame=white_mask


# t = pyt.image_to_string(frame, config='--psm 13')
# t = pyt.image_to_string(frame, config='--psm 7 -c tessedit_char_whitelist=0123456789')
# print('>>'+t)






    










# # Open the webcam
# cap = cv.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    

#         # d1=segread(i1)
#         d2=segread(i2)
#         # d3=segread(i3)
#         # print(d2)

#         # print('>'+str(d1)+str(d2)+str(d3))

        
#         # i1 = cv.cvtColor(i1, cv.COLOR_BGR2RGB)
# #         t = pyt.image_to_string(i1, lang='eng', config='--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')
# #         print('>'+t)
# # # ///////////////////////////////////////////
#         # kernel = np.ones((5,5), np.uint8);

#         # # dilate
#         # i1 = cv.dilate(i1, kernel, iterations=1);

#         # erode
#         # for a in range(iters):
#         #     thresh = cv.erode(thresh, kernel);
         

#     # Display the frame
#     cv.imshow('Webcam', dup)

#     # Check if 'q' key is pressed to exit
#     if cv.waitKey(1) == ord('q'):
#         break

# # Release the webcam and close the windows
# cap.release()

