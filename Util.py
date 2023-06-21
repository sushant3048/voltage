import cv2
import pytesseract

cap = cv2.VideoCapture(0)
def capture():
    _,img = cap.read()
    return img

def crop(img, rec):
    # bounds is crop rectangle
    x,y,w,h=rec
    cropped = img[y:y+h, x:x+w]
    return cropped


img=capture()
print(img.shape)