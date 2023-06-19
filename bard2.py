import cv2
import pytesseract
import numpy as np
import re

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue color range
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply morphology operations to enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop the largest contour (blue rectangle)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = frame[y:y+h, x:x+w]

        # Convert cropped image to grayscale
        cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply additional image processing techniques (e.g., thresholding, denoising, etc.) to enhance OCR accuracy
        # ...

        # Apply OCR to the preprocessed image
        result = pytesseract.image_to_string(cropped_gray, config='--psm 7')

        # Filter out non-numeric characters
        numeric_result = re.sub(r'[^0-9]', '', result)

        # Print the extracted numeric value
        print("Extracted value:", numeric_result)

        # Draw the bounding rectangle on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
