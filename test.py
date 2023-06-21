import numpy as np
import cv2 as cv

inp = cv.imread('media/210.jpg')
hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)

# Define the lower and upper blue thresholds
lower_blue = np.array([95, 0, 0])
upper_blue = np.array([105, 255, 255])

# Create a mask for blue pixels
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Find contours in the mask
contours, _ = cv.findContours(
    mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

dims = inp.shape
print('dims', dims)
print(dims[0])

myimg = mask.copy()
for a in range(dims[0]):
    for b in range(dims[1]):
       print(inp[a][b])
       
print('mask shape', mask.shape)
