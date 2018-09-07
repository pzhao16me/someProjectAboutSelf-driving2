
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
from tools import lane_finding_pipeline


class CameraCalibration():
    """A class that contains the calibration data gathered from input calibration images

    Attributes:
        ym_per_pix (int): Meters per pixel in the y direction
        xm_per_pix (int): Meters per pixel in the x direction
        chessboardImagesPath (str): The path to the calibration chessboard images
        perspectiveTransformCalibrationFilePath (str): Path to the image used for perspective transform calculation
        nx (int): number of cols in the calibration chessboards
        ny (int): number of rows the calibration chessboards
        mtx(array): camera matrix
        dist(array): distortion coefficients
        M (array): transformation matrix
        Minv (array): inverse transformation matrix
    """

    def __init__(self, chessboardImagesPathIn, nxIn, nyIn, perspectiveTransformCalibrationFilePathIn):

        # Constants
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension, taken from course materials
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension, taken from course materials

        # Input params
        self.chessboardImagesPath = chessboardImagesPathIn
        self.perspectiveTransformCalibrationFilePath = perspectiveTransformCalibrationFilePathIn
        self.nx = nxIn
        self.ny = nyIn

        # Stored calibration data
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None

        # Initialize Cal data on creation
        self.process_calibration_data()

    def process_calibration_data(self):
        """Executes both calibration calculations"""
        self.calc_distortion()
        self.calc_perspective()

    def calc_distortion(self):
        """Calculates the distortion coefficients using the input calibration images"""

        # Obj points and img points for calibration
        objpoints = []
        imgpoints = []
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        # Read input chessboard images
        images = glob.glob(os.path.join(self.chessboardImagesPath, "*.jpg"))
        for fname in images:
            img = mpimg.imread(fname)

            # Convert to greyscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        # Calculate distortion coefficients
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                     imgpoints, gray.shape[::-1],
                                                                     None, None)

    def calc_perspective(self, plot_lane_image=False):
        """Calculates the perpsective trainform matrix.

        First executes the original lane finding pipeline on a simple straight
        lane image, and then maps that to vertically oriented lines in an overhead perspective.
        """
        img = mpimg.imread(self.perspectiveTransformCalibrationFilePath)

        ### Use original line finding to find the lanes on the perspective calibration image
        imshape = img.shape
        left_cutoff = 100
        right_cutoff = 50
        bottom_cutoff = 50
        top_cutoff = imshape[0] / 2 + 125
        endpoint_lane_width = 350
        region_mask = np.array([[(left_cutoff, imshape[0] - bottom_cutoff),
                                 (imshape[1] / 2 - endpoint_lane_width / 2, top_cutoff),
                                 (imshape[1] / 2 + endpoint_lane_width / 2, top_cutoff),
                                 (imshape[1] - right_cutoff, imshape[0] - bottom_cutoff)]],
                               dtype=np.int32)
        lines, lane_image = lane_finding_pipeline(img, region_mask)

        # Show the marked image if requested
        if plot_lane_image:
            plt.imshow(lane_image)

        # Set the source and destination points to transform the image to overhead
        src = np.float32([lines[0][0],
                          lines[0][1],
                          lines[1][0],
                          lines[1][1]])
        dst = np.float32([[lines[0][0][0] - left_cutoff, imshape[0]],
                          [lines[0][0][0] - left_cutoff, 0],
                          [lines[1][1][0] + right_cutoff, 0],
                          [lines[1][1][0] + right_cutoff, imshape[0]]])

        print("src points:", src)
        print("/t")
        print("dst points:", dst)

        # Calculate perspective transform matrices
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def undistort_image(self, InputImage):
        """Undistort image using internal distortion coefficients"""
        return cv2.undistort(InputImage, self.mtx, self.dist, None, self.mtx)

    def warp_to_overhead_perspective(self, InputImage):
        """Warp to overhead using transform matrix"""
        return cv2.warpPerspective(InputImage, self.M,
                                   (InputImage.shape[1], InputImage.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def warp_to_original_perspective(self, InputImage):
        """Warp to original using inverse transform matrix"""
        return cv2.warpPerspective(InputImage, self.Minv,
                                   (InputImage.shape[1], InputImage.shape[0]),
                                   flags=cv2.INTER_LINEAR)