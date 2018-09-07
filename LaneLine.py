import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os

import queue


class LaneLine():
    """A class that contains the characteristics of each line detection.

    Attributes:
        cameraCal (CameraCalibration):  Camera calibration data

        size_of_history (int): amount of historical fits to store
        min_detections (int): minimum number of accepted fits before lane can be considered detected
        dropped_frames_threshold (int): Maximum amount of dropped frames before lane state is reset
        radius_thresholds (array): min and max thresholds for radius in meters
        lane_pos_threshold (float): maximum threshold for distance from lane center
        coefficient_threshold_score (float): maximum outlier score theshold

        detected (bool): Was the line detected in the last iteration?
        dropped_frames (int): Amount of subsequent frame drops
        recent_xfitted (queue): x values of the last "size_of_history" fits of the line
        recent_fitted (queue): polynomial coefficient of the last "size_of_history" fits of the line
        bestx (int): average x values of the fitted line over the last "size_of_history" iterations
        best_fit (int): polynomial coefficients averaged over the last "size_of_history" iterations
        current_fit (int): polynomial coefficients for the most recent fit
        current_fitx (int): x values for the most recent fit
        radius_of_curvature (int): radius of curvature of the line in some units
        line_base_pos (int): distance in meters of vehicle center from the line
        diffs (array): difference in fit coefficients between last and new fits
        allx (array): x values for detected line pixels
        ally (array): y values for detected line pixels
    """
    dropped_frames: int

    def __init__(self, cameraCalIn):

        # Input calibration
        self.cameraCal = cameraCalIn

        # Hyper params
        self.size_of_history = 10
        self.min_detections = 10
        self.dropped_frames_threshold = 10
        self.radius_thresholds = [100, 100000]
        self.lane_pos_threshold = 3.0
        self.coefficient_threshold_score = 1.2

        # Lane detection state information
        self.detected = False
        self.dropped_frames = 0
        self.recent_xfitted = queue.Queue(maxsize=self.size_of_history)
        self.recent_fitted = queue.Queue(maxsize=self.size_of_history)
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.current_fitx = None
        self.radius_of_curvature = 0
        self.line_base_pos = 0
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

    def add_new_lines(self, x, y, image_height=720, image_width=1280):
        """Add new lane line data into the pipeline"""
        self.allx = x
        self.ally = y

        # Drop frame if no lane pixels are detected
        if (x.size == 0 or y.size == 0):
            self.dropped_frames += 1
            self.check_drop_limit()
            return

            # Fit a second order polynomial to each using `np.polyfit`
        self.current_fit = np.polyfit(y, x, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image_height - 1, image_height)
        try:
            self.current_fitx = self.current_fit[0] * ploty ** 2 + self.current_fit[1] * ploty + self.current_fit[2]
        except TypeError:
            # Drop frame if fit fails
            print('The function failed to fit a line!')
            self.dropped_frames += 1
            self.check_drop_limit()
            return

            # Calc fit difference
        if self.detected:
            self.diffs = self.best_fit - self.current_fit

        # Calculate fit in Meters
        fit_m = np.polyfit(self.ally * self.cameraCal.ym_per_pix, self.allx * self.cameraCal.xm_per_pix, 2)

        # Calc radius in meters
        self.radius_of_curvature = ((1 + (
                    2 * fit_m[0] * image_height * self.cameraCal.ym_per_pix + fit_m[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_m[0])

        # Calc lane distatnce from center.

        # First need to extrapolate lowest x point.
        line_base_x = fit_m[0] * ((image_height * self.cameraCal.ym_per_pix) ** 2) + fit_m[1] * (
                    image_height * self.cameraCal.ym_per_pix) + fit_m[2]

        # Then subtract from center
        self.line_base_pos = np.absolute(image_width * self.cameraCal.xm_per_pix / 2 - line_base_x)

        # Update fit averages and throw up bad data
        self.update_best_fit()

        # Verify we haven't exceeded the drop limit
        self.check_drop_limit()

    def reset_state(self):
        """Reset lane state"""
        self.detected = False
        self.droppedFrames = 0
        self.recent_xfitted = queue.Queue(maxsize=self.size_of_history)
        self.recent_fitted = queue.Queue(maxsize=self.size_of_history)
        self.bestx = None
        self.best_fit = None

    def update_best_fit(self):
        """Updates lane state with new input data"""
        ## Validate vals if we have a line to check against
        if ((self.line_base_pos > self.lane_pos_threshold) or
                (self.radius_of_curvature < self.radius_thresholds[0]) or
                (self.radius_of_curvature > self.radius_thresholds[1])):
            self.dropped_frames += 1
            return

        if self.detected:
            # Calculate outlier score based on Median Absoulte Deviation
            # See https://en.wikipedia.org/wiki/Median_absolute_deviation
            coeff = np.array(list(self.recent_fitted.queue))
            median = np.median(coeff, axis=0)
            diff = np.absolute(coeff - median)
            med_abs_deviation = np.median(diff, axis=0)
            score = 0.6745 * self.diffs / med_abs_deviation
            if (score.any() > self.coefficient_threshold_score):
                self.dropped_frames += 1
                return

        # If we got to this point, we are going to keep the frame.
        self.dropped_frames = 0

        # Make sure we don't overflow
        if self.recent_xfitted.full():
            self.recent_xfitted.get()
            self.recent_fitted.get()

            # Add data to queues
        self.recent_xfitted.put(self.current_fitx)
        self.recent_fitted.put(self.current_fit)

        # Check if we've reached a detection min
        if self.recent_xfitted.qsize() >= self.min_detections:
            self.detected = True

        # Caclulate average
        if self.recent_xfitted.qsize() == 1:
            self.bestx = self.current_fitx
            self.best_fit = self.current_fit
        else:
            self.bestx = np.average(np.array(list(self.recent_xfitted.queue)), axis=0)
            self.best_fit = np.average(np.array(list(self.recent_fitted.queue)), axis=0)

    def check_drop_limit(self):
        """Check for a dropped frames limit, and reset if necessary"""
        if self.dropped_frames >= self.dropped_frames_threshold:
            self.reset_state()