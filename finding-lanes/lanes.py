import cv2
import numpy as np
import matplotlib.pyplot as plt

def average_slope_intercept(image, lines):
    left_fit = [] # Cooridnates of the lines on the left
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = lines.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1) # gives slope and y intercept for the coords.
        slope = parameters[0]
        intercept = parameters[1]
        # To determine the lines on the left and right, note that the lines on the left will be slanted
        # a little to the right. Same is for right lines(Slanted to left). That's how lanes are.
        # Since y is reversed in a computer's pixel coords, right lines will have positive slope.
        
        

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Step 1 convert image to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Step 2 Smoothen the image with a Gaussian Filter.
    canny = cv2.Canny(blur, 50, 150) # Step 3 Use canny edge detection to detect edges.
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # Computing this will only show the region of interest in canny image.
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(image)
cropped_image = region_of_interest(canny_image)
# Detect lines in the image.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
# Display lines from hough transform over a black image
line_image = display_lines(image, lines)
# Combine the black image with lines with the original image.
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('window', combo_image)
cv2.waitKey(0)
# Using matplotlib to display the image helps us to determine points for region of interest.
# plt.imshow(canny)
# plt.show()


