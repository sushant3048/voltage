# pip install opencv-python
# pip install pytesseract
# Note that pytesseract requires the Tesseract OCR engine to be installed on your system. You can download it from the official website: https://github.com/tesseract-ocr/tesseract

# The program continuously reads frames from the webcam, performs OCR on each frame using pytesseract, and extracts the numeric text. It then prints the numeric text on the console. Optionally, you can display the webcam feed with bounding boxes by uncommenting the cv2.imshow('Webcam', frame) line.


import cv2
import pytesseract

def scan_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Perform OCR on the frame
        text = pytesseract.image_to_string(frame, config='--psm 6 digits')

        # Filter numeric text
        numeric_text = ''.join(filter(str.isdigit, text))

        # Print numeric text on console
        print(numeric_text,'/////', text)

        # Display the frame with bounding boxes (optional)
        cv2.imshow('Webcam', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam scanning program
scan_webcam()
