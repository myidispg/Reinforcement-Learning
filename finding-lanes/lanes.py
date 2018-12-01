import cv2
import numpy as np
import matplotlib.pyplot as plt

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
canny = canny(image)
cropped_image = region_of_interest(canny)
# Detect lines in the image.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# Display lines from hough transform over a black image
line_image = display_lines(image, lines)
# Combine the black image with lines with the original image.
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('window', combo_image)
cv2.waitKey(0)
# Using matplotlib to display the image helps us to determine points for region of interest.
# plt.imshow(canny)
# plt.show()


