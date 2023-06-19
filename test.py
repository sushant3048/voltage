import numpy as np
import cv2
lowest= np.uint8([[[238,142,1 ]]])
highest= np.uint8([[[249,190,20 ]]])
lowest_hsv = cv2.cvtColor(lowest,cv2.COLOR_BGR2HSV)
highest_hsv = cv2.cvtColor(highest,cv2.COLOR_BGR2HSV)

print(lowest_hsv,highest_hsv)

# rgb(8,166,244)
# rgb(22,181,247)
# rgb(8,176,249)
# rgb(3,158,248)
# rgb(3,147,246)
# rgb(3,142,245)
# rgb(1,144,248)
# rgb(3,155,248)
# rgb(67,190,247)
# rgb(7,173,238)