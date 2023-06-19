import cv2
import numpy as np
import pytesseract as pyt

def find_bluest_object(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper blue thresholds
    lower_blue = np.array([95, 0, 0])
    upper_blue = np.array([105, 255, 255])

    # Create a mask for blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Return the bounding rectangle of the bluest object
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        return (x, y, w, h)
    else:
        return None


def crop(img,bounds):
    (x1, y1),(x2,y2)=bounds
    cropped=img[y1:y2,x1:x2]
    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    (th, bw) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return bw

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    dup=frame.copy()

    # Find the bluest object in the frame
    rectangle = find_bluest_object(frame)

    if rectangle is not None:
        # Draw a rectangle around the bluest object
        x, y, w, h = rectangle
        
        #tightening the rectangle with experimental values
        x=int(x+w*0.05)
        w=int(w*0.9)
        y=int(y+h*0.05)
        h=int(h*0.95)

        # slicing into 3 parts for each digit.
        wd=int(w/3)
        x1=x+wd
        x2=x1+wd

        # cords of each digit
        r1=[(x, y), (x1, y+h)]
        r2=[(x1, y), (x2, y+h)]
        r3=[(x2, y), (x+w, y+h)]

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.rectangle(dup, *r1, (0, 255, 255), 2)
        cv2.rectangle(dup, *r2, (0, 255, 255), 2)
        cv2.rectangle(dup, *r3, (0, 255, 255), 2)

        # idenfity the digit
        i1=crop(frame,r1)
        cv2.imshow('first',i1)
        img_rgb = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
        t=pyt.image_to_string(img_rgb)
        print('>>'+t)


    # Display the frame
    cv2.imshow('Webcam', dup)

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
