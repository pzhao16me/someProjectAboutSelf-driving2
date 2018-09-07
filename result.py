from  CameraCalibration import  CameraCalibration
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# print("current:", os.getcwd())

cameraCal = CameraCalibration("camera_cal",9,6, os.path.join("test_images","straight_lines2.jpg"))

# Test cal data on simple straight line image
straight_lines = mpimg.imread(os.path.join("test_images","straight_lines2.jpg"))
# First undistort the image
undist = cameraCal.undistort_image(straight_lines)

fig = plt.figure(figsize=(20,10))
plt.axis('off')
ax = fig.add_subplot(1,2,1)
ax.imshow(straight_lines)
ax.set_title("Original")
ax = fig.add_subplot(1,2,2)
ax.imshow(undist)
ax.set_title("Undistorted")
