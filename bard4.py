import cv2
import numpy as np
import pygame
from pygame.locals import *

# Function to detect 7-segment digits in a frame
def detect_digits(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the digits
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the digits
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    # Iterate over the contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filter out small contours
        if w > 10 and h > 10:
            digits.append((x, y, w, h))

    return digits

# Initialize Pygame
pygame.init()

# Set the display dimensions
display_width = 800
display_height = 600

# Create the Pygame display surface
game_display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('7-Segment Digit Detection')

# Function to render smooth text on the display surface
def render_text(text, x, y, size):
    font = pygame.font.Font(None, size)
    rendered_text = font.render(text, True, (255, 255, 255))
    game_display.blit(rendered_text, (x, y))

# Main program loop
def main():
    # Open the video capture
    cap = cv2.VideoCapture(0)

    # Set the video capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Detect 7-segment digits in the frame
        digits = detect_digits(frame)

        # Display the frame in the Pygame window
        game_display.blit(pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))

        # Render text on top of the detected digits
        for digit in digits:
            (x, y, w, h) = digit
            render_text("Digit", x, y, 20)

        # Update the Pygame display
        pygame.display.update()

        # Check for quit event
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                cap.release()
                return

if __name__ == '__main__':
    main()
