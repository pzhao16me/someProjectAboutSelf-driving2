import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
import queue


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def clean_lines(lines, region_of_interest_verticies):
    """Cleans up the list of lines and calulates
    a single extrapolated left and right lane line
    """
    # Remove anything with a slope to close to 0 (horizontal lines)
    pruned_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1))
            if (slope > 0.4 or slope < -0.4):
                pruned_lines.append(line)

    lines = np.array(pruned_lines)

    # Separate by slope into left and right
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1))
            if slope < 0:
                left_lines.append([x1, y1, x2, y2])
            else:
                right_lines.append([x1, y1, x2, y2])

    # Calc extrapolated left line.  Pull y values from region of interest
    extraoplated_x_left = extrapolate_line(np.average(np.array(left_lines), axis=0),
                                           region_of_interest_verticies[0][1],
                                           region_of_interest_verticies[1][1])

    left_line = [(extraoplated_x_left[0], region_of_interest_verticies[0][1]),
                 (extraoplated_x_left[1], region_of_interest_verticies[1][1])]

    # Calc extrapolated right line.  Pull y values from region of interest
    extraoplated_x_right = extrapolate_line(np.average(np.array(right_lines), axis=0),
                                            region_of_interest_verticies[2][1],
                                            region_of_interest_verticies[3][1])

    right_line = [(extraoplated_x_right[0], region_of_interest_verticies[2][1]),
                  (extraoplated_x_right[1], region_of_interest_verticies[3][1])]

    return [left_line, right_line]


def extrapolate_line(line, y_boundary_1, y_boundary_2):
    """Extrapolates a input line to the given y boundaries"""
    # Calculate slope
    try:
        slope = ((line[3] - line[1]) / (line[2] - line[0]))
    except IndexError:
        # For some reason, the "Challege" video causes HoughLinesP to produce some "empty" lines...
        return [-1, -1]

        # Calculate x1 and x2 using modified slope forumla x1 = x2 - (y2 -y1)/slope
    x1 = int(line[0] - (line[1] - y_boundary_1) / slope)
    x2 = int(line[2] - (line[3] - y_boundary_2) / slope)

    return [x1, x2]


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, region_of_interest_verticies):
    """Returns an image with hough lines drawn."""

    # Get all the lines from the input image
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # Create a base image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Clean the lines and draw them on the image
    cleaned_lines = clean_lines(lines, region_of_interest_verticies)
    cv2.line(line_img, cleaned_lines[0][0], cleaned_lines[0][1], [255, 0, 0], 5)
    cv2.line(line_img, cleaned_lines[1][0], cleaned_lines[1][1], [255, 0, 0], 5)

    # Draw the region of interest
    cv2.line(line_img, tuple(region_of_interest_verticies[0]), tuple(region_of_interest_verticies[1]), [0, 255, 0], 7)
    cv2.line(line_img, tuple(region_of_interest_verticies[1]), tuple(region_of_interest_verticies[2]), [0, 255, 0], 7)
    cv2.line(line_img, tuple(region_of_interest_verticies[2]), tuple(region_of_interest_verticies[3]), [0, 255, 0], 7)
    cv2.line(line_img, tuple(region_of_interest_verticies[3]), tuple(region_of_interest_verticies[0]), [0, 255, 0], 7)

    return cleaned_lines, line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def lane_finding_pipeline(img, region_mask):
    """Full original lane finding pipeline.
    Takes in an image and region mask,
    and returns the two lane lines and a
    marked image with the lines and region mask overliad"""

    # Use Canny edge detection on graysclaed image
    guassian_kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    edges = canny(gaussian_blur(grayscale(img), guassian_kernel_size), low_threshold, high_threshold)

    # Mask area of interest (Use image size to calculate this to account for image/video sizes)
    masked_image = region_of_interest(edges, region_mask)

    # Apply Hough transform and draw wighted lines
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 2
    lines, line_img = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap, region_mask[0])

    # Overlay the line image with the original
    line_img = weighted_img(line_img, img)
    return lines, line_img