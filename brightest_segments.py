import cv2
import pytesseract
import numpy as np
import re

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image to extract the brightest regions
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area (brightest object)
    # cropped_width=170
    # cropped_height=95
    k=95
    o=10
    w=50
    x1=o+w
    x2=o+2*w
    e=o+3*w-10
    recs=[[(o,0),(x1,k)],[(x1,0),(x2,k)],[(x2,0),(e,k)]]

    if len(contours) > 0:
        brightest_contour = max(contours, key=cv2.contourArea)
        
        # Draw a bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(brightest_contour)
        # print(w,h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = frame[y:y+h, x:x+w]
        cv2.rectangle(cropped, recs[0][0], recs[0][1], (0, 0, 255), 1)
        cv2.rectangle(cropped, recs[1][0], recs[1][1], (0, 0, 255), 1)
        cv2.rectangle(cropped, recs[2][0], recs[2][1], (0, 0, 255), 1)
        cv2.imshow('cropped', cropped)
        first=cropped[:, o:x1]
        second=cropped[:, x1:x2]
        third=cropped[:, x2:e]
        # cv2.imshow('first', first)
        # cv2.imshow('second', second)
        # cv2.imshow('third', third)

    
    # Display the resulting frame
    cv2.imshow('Brightest Object', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
