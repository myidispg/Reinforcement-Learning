import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coords(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2]) 

def average_slope_intercept(image, lines):
    """
    This function is used to combine multiple lines on a single lane into a single consistent lane. 
    This works by dividing all lines into left and right fit based on the slope and taking their average. 
    """
    left_fit = [] # Cooridnates of the lines on the left
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1) # gives slope and y intercept for the coords.
        slope = parameters[0]
        intercept = parameters[1]
        # To determine the lines on the left and right, note that the lines on the left will be slanted
        # a little to the right. Same is for right lines(Slanted to left). That's how lanes are.
        # Since y is reversed in a computer's pixel coords, right lines will have positive slope.
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coords(image, left_fit_average)
    right_line = make_coords(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Step 1 convert image to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Step 2 Smoothen the image with a Gaussian Filter.
    canny = cv2.Canny(blur, 50, 150) # Step 3 Use canny edge detection to detect edges.
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # Computing this will only show the region of interest in canny image.
    return masked_image

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# # Detect lines in the image.
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# # Display lines from hough transform over a black image
# line_image = display_lines(image, averaged_lines)
# # Combine the black image with lines with the original image.
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('window', combo_image)
# cv2.waitKey(0)
# ---Keep this portion commented unless you want to see the coords in the image.
# Using matplotlib to display the image helps us to determine points for region of interest.
# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    # Detect lines in the image.
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    # Display lines from hough transform over a black image
    line_image = display_lines(frame, averaged_lines)
    # Combine the black image with lines with the original image.
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('window', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()