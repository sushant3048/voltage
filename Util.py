import cv2
import pytesseract

cap = cv2.VideoCapture(0)
def capture():
    _,img = cap.read()
    return img

def crop(img, bounds):
    # bounds is crop rectangle
    (x1, y1), (x2, y2) = bounds
    cropped = img[y1:y2, x1:x2]
    return cropped


