    # Crop the largest contour (blue rectangle)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = frame[y:y+h, x:x+w]