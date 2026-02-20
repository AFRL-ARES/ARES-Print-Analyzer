# -*- coding:utf-8 -*-
###
# File: /tools/camera_calibraiton.py
# Project: ARES-Print-Analyzer
# Created Date: Friday, February 20th 2026, 12:36:01 pm
# Author(s): Arthur W. N. Sloan
# -----
# MIT License
# 
# Copyright (c) 2025 AFRL-ARES
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
###

import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
import json

"""
Camera Calibration Tool for ARES Print Analyzer
This module performs camera calibration using a checkerboard pattern to determine
camera intrinsic parameters (camera matrix and distortion coefficients). It supports
both standard and fisheye lens calibration models. It is designed around the use of the 9x6 open CV 
calibration pattern (https://github.com/opencv/opencv/blob/4.x/doc/pattern.png) printed such that the squares are 22 mm wide.
See https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html for more info on acquiring calibration images.

Configuration:
    image_dir (Path): Directory containing calibration images (*.jpg) and output location
    square_size (int): Size of checkerboard squares in millimeters
    fisheye (bool): If True, uses fisheye calibration model; otherwise uses standard model

Output:
    - Prints camera matrix and distortion coefficients to console
    - Saves calibration data to 'camera_calibration.json' in image_dir
    - Displays comparison plot of original and undistorted images
"""
#  Calibration type and image source directory
image_dir = Path('<Path/to/your/calibraiton/images/folder>') # This is also where the camera_calibration.json file will be saved
square_size = 22 # in mm 
fisheye = True 


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW

obj_points = [] # Points in real world space
img_points = [] # Points in image plane

pattern_size = (9,6)
obj_p = np.zeros((1, pattern_size[1]*pattern_size[0], 3), np.float32)
obj_p[0,:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)*square_size

file_list = list(image_dir.glob('*.jpg'))
for img_file in file_list:
    img = cv.imread(str(img_file))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, 
                                            pattern_size, 
                                            cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE) # type: ignore
    if ret:
        obj_points.append(obj_p)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners2)

if fisheye:
    ret, K, D, rvecs, tvecs = cv.fisheye.calibrate(obj_points, img_points, gray.shape[::-1], None, None, None, None, calibration_flags, (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) # type: ignore
else:
    ret, K, D, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None) # type: ignore


print(" Camera matrix:")
print(K)

print("\n Distortion coefficient:")
print(D)

json_path = image_dir / 'camera_calibration.json'
print(f"Saving camera calibration to: {str(json_path)}")
data={'camera_matrix':K.tolist(), 'distortion_coefficients':D.tolist(),'fisheye':True}
with open(str(json_path), 'w') as f:
    json.dump(data, f)


# Plot a random calibration image along side its undistored version
img_file = file_list[np.random.randint(0,len(file_list))]
img = cv.imread(str(img_file))

DIM = img.shape[:2][::-1]
if fisheye:
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)
else:
    K_new, roi = cv.getOptimalNewCameraMatrix(K, D, DIM, 1, DIM)
    map1, map2 = cv.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)

dst = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
ax[1].set_title('Undistorted Image')
ax[1].axis('off')
plt.show()


