import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Step 1 convert image to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Step 2 Smoothen the image with a Gaussian Filter.
    canny = cv2.Canny(blur, 50, 150) # Step 3 Use canny edge detection to detect edges.
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(image)
# Using matplotlib to display the image helps us to determine points for region of interest.
cv2.imshow('window', canny)
cv2.waitKey(0)
# plt.imshow(canny)
# plt.show()


