import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2
from  CameraCalibration import  CameraCalibration



cameraCal = CameraCalibration("camera_cal",9,6, os.path.join("test_images","straight_lines2.jpg"))

class CameraImage():
    """A class that contains the characteristics of each camera image being used for lane detection.

    Attributes:
        img (image): The current image
        orginalImage (image): The original image passed into this class
        cal (CameraCalibration):  Camera calibration data
        improved_thresh(boolean): added later to improve the threshold
        distorted(boolean): whether or not the image is distorted
        thresholds_applied(boolean): whether or not the imaging thresholds have been applied
        transformed(boolean): whether or not a perspective transform has been applied
    """

    def __init__(self, imageIn, cameraCal, improvedThreshIn=False):

        # The current image
        self.img = imageIn

        # Input information
        self.originalImage = imageIn
        self.cal = cameraCal
        self.improved_thresh = improvedThreshIn

        # State information
        self.distorted = True
        self.thresholds_applied = False
        self.transformed = False

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Applies gradient Sobel threshold in the specified direction"""
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Use inclusive thresholds
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return the result
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        """Applies a gradient magnitude threshold"""
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """Applies a directional gradient theshold"""
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def hls_select(self, chan, thresh=(0, 255)):
        """Applies a theshold on the given hls channel"""
        binary_output = np.zeros_like(chan)
        binary_output[(chan > thresh[0]) & (chan <= thresh[1])] = 1
        return binary_output

    def hls(self):
        """Converts RGB image to HLS"""
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)

    def gray(self):
        """Converts RGB image to greyscale"""
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def color_select(self, color_image, thresh=[(0, 0), (0, 0), (0, 0)]):
        """Applies color thesholds on the given image and returns the binary result"""

        # Extract each color channel
        r = color_image[:, :, 0]
        g = color_image[:, :, 1]
        b = color_image[:, :, 2]

        # Combine into single channel
        binary_mins = np.zeros_like(r)
        binary_mins[(r >= thresh[0][0]) & (r <= thresh[0][1])
                    & (g >= thresh[1][0]) & (g <= thresh[1][1])
                    & (b >= thresh[2][0]) & (b <= thresh[2][1])] = 1
        return binary_mins

    def undistort_image(self):
        """Undistort image using calibration data"""
        if self.distorted:
            self.img = cameraCal.undistort_image(self.img)
            self.distorted = False

    def apply_thresholds(self, plot=False):
        """Apply full set of chosen thresholds"""
        if not self.thresholds_applied:
            # Select low hue and high saturation values
            hls = self.hls()
            h_channel = hls[:, :, 0]
            s_channel = hls[:, :, 2]
            h_select = self.hls_select(h_channel, thresh=(0, 30))
            s_select = self.hls_select(s_channel, thresh=(100, 255))

            if self.improved_thresh:
                # Color thresholding to attempt to gather pure yellow/white sections
                yellow = self.color_select(self.img, thresh=[(140, 255), (140, 255), (0, 120)])
                white = self.color_select(self.img, thresh=[(200, 255), (200, 255), (200, 255)])
                color_select = np.zeros_like(yellow)
                color_select[(yellow == 1) | (white == 1)] = 1

                # Gradients on Greyscale using suggested thresholds to find lane lines
            sobelx = self.abs_sobel_thresh(self.gray(), orient='x', thresh=(20, 100))
            sobeldir = self.dir_threshold(self.gray(), sobel_kernel=15, thresh=(0.5, 1.3))
            sobel_combined = np.zeros_like(sobeldir)
            sobel_combined[(sobelx == 1) & (sobeldir == 1)] = 1

            if not self.improved_thresh:
                # Combine all thresholds as such:
                # *  low hue and high saturation
                # *  Selected gradients
                combined_binary = np.zeros_like(h_select)
                combined_binary[((h_select == 1) & (s_select == 1)) |
                                (sobel_combined == 1)] = 1

                # Plot results if requested
                if (plot):
                    plt.subplot(2, 2, 1)
                    plt.imshow(sobel_combined, cmap="gray")
                    plt.title("sobel_combined")
                    plt.subplot(2, 2, 2)
                    plt.imshow(h_select, cmap="gray")
                    plt.title("h_select")
                    plt.subplot(2, 2, 3)
                    plt.imshow(s_select, cmap="gray")
                    plt.title("s_select")
                    plt.subplot(2, 2, 4)
                    plt.imshow(combined_binary, cmap="gray")
                    plt.title("combined_binary")
                    plt.show()
            else:
                # Improved thresholding after reviewing challenge video
                # Combine all thresholds as such:
                # *  low hue and high saturation
                # *  Yellow or white pixels
                # *  Low hue and selected gradients
                combined_binary = np.zeros_like(h_select)
                combined_binary[((h_select == 1) & (s_select == 1)) |
                                (color_select == 1) |
                                ((h_select == 1) & (sobel_combined == 1))] = 1

                # Plot results if requested
                if (plot):
                    plt.subplot(3, 2, 1)
                    plt.imshow(sobel_combined, cmap="gray")
                    plt.title("sobel_combined")
                    plt.subplot(3, 2, 2)
                    plt.imshow(yellow, cmap="gray")
                    plt.title("yellow")
                    plt.subplot(3, 2, 3)
                    plt.imshow(white, cmap="gray")
                    plt.title("white")
                    plt.subplot(3, 2, 4)
                    plt.imshow(h_select, cmap="gray")
                    plt.title("h_select")
                    plt.subplot(3, 2, 5)
                    plt.imshow(s_select, cmap="gray")
                    plt.title("s_select")
                    plt.subplot(3, 2, 6)
                    plt.imshow(combined_binary, cmap="gray")
                    plt.title("combined_binary")
                    plt.show()

            # Update image and state
            self.img = combined_binary
            self.thresholds_applied = True

    def perspective_transform(self):
        """Transform image to overhead perspective"""
        if not self.transformed:
            self.img = cameraCal.warp_to_overhead_perspective(self.img)
            self.transformed = True

    def apply_full_pipeline(self):
        """Convenince function: Apply each step in the pipeline"""
        self.undistort_image()
        self.apply_thresholds()
        self.perspective_transform()