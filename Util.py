import cv2
import pytesseract

cap = cv2.VideoCapture(0)
def capture():
    _,img = cap.read()
    return img


def pyread1(frame):
    text = pytesseract.image_to_string(frame, config='--psm 6 digits')
    return text

def pyread2(frame):
    text = pytesseract.image_to_string(frame)
    return text