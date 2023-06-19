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

        # Apply image preprocessing techniques (e.g., thresholding, denoising, sharpening) to enhance OCR accuracy
        cropped_gray = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cropped_gray = cv2.medianBlur(cropped_gray, 3)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        cropped_gray = cv2.filter2D(cropped_gray, -1, kernel)

        # Divide the cropped image into three equal segments
        segment_width = int(w / 3)
        segments = [cropped_gray[:, i*segment_width:(i+1)*segment_width] for i in range(3)]

        extracted_text = ''

        # Process each segment individually and extract numeric characters
        for segment in segments:
            # Find contours of individual characters
            character_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours from left to right
            character_contours = sorted(character_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            # Extract each character and apply OCR
            for contour in character_contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                character_image = segment[cy:cy+ch, cx:cx+cw]

                # Apply OCR to the character
                character_result = pytesseract.image_to_string(character_image, config='--psm 10 -c tessedit_char_whitelist=0123456789')

                # Filter out non-numeric characters
                numeric_character_result = re.sub(r'[^0-9]', '', character_result)

                # Add the numeric character to the extracted text
                extracted_text += numeric_character_result

        # Validate if three digits are present
        # if len(extracted_text) == 3 and extracted_text.isdigit():
        #     # Print the extracted numeric value
        print("Extracted value:", extracted_text)

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
