#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/analysis/get_contours.py
# Project: ARES-Print-Analyzer
# Created Date: Friday, October 24th 2025, 10:48:11 am
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

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def downselect_contours(contours: tuple,
                        center: np.ndarray) -> tuple:
    # downselect the contours based on the location relative to the 
    # center of onbject predicted from the CV step.
    # to account for objects where the center of the bouding box may not be within
    # the object, we use the convex hull, first searching for hull that contains the center
    # and if that fails, finding the contour that is closes to the hull
    dists = []
    for c in contours:
        h = cv.convexHull(c)
        dist = cv.pointPolygonTest(h,(int(center[0]),int(center[1])),True)
        dists.append(dist)
    dists = np.array(dists)

    if np.any(dists > 1): # positive distance means it is inside the contour
        c_arg = np.argwhere(dists > 1)
        c_arg = c_arg.flatten()[0]
    else: # if no hull contains the point, get the closest (least nevative distance)
        c_arg = np.argmax(dists)

    return (contours[c_arg])

def get_contours(experimental_image: np.ndarray,
                 synthetic_image: np.ndarray,
                 object_center: np.ndarray,
                 debug: bool = False) -> tuple[np.ndarray,np.ndarray]:
    # Returns the opencv contours of the experimental and synthetic object
    img_W = experimental_image.shape[1]
    img_H = experimental_image.shape[0]

    # Convert images to grayscale
    syn_gray = cv.cvtColor(synthetic_image,cv.COLOR_BGR2GRAY)
    exp_gray = cv.cvtColor(experimental_image,cv.COLOR_BGR2GRAY)


    # Use Otsu's method to estimate the right global threshold value and then tweak it a bit for the actual
    syn_ret , _ = cv.threshold(syn_gray, 0, 255, cv.THRESH_OTSU)
    exp_ret, _ = cv.threshold(exp_gray, 0, 255, cv.THRESH_OTSU)
    _ , syn_mask = cv.threshold(syn_gray, syn_ret*1.1, 255, cv.THRESH_BINARY)
    _ , exp_mask = cv.threshold(exp_gray, exp_ret*1.1, 255, cv.THRESH_BINARY)

    # Find the contours of all white (True) blobs in the mask images.
    # This can return multiple contours, but thanks to the CV pose estimation we know where the 
    # object should be in image coordinates

    exp_cont,_ = cv.findContours(syn_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(exp_cont) > 0:
        experimental_contour = downselect_contours(exp_cont, object_center)
    else:
        raise Warning("Could Not find any experimental contours")

    syn_cont,_ = cv.findContours(syn_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(syn_cont) > 0:
        synthetic_contour = downselect_contours(syn_cont,object_center)
    else:
        raise Warning("Could Not find any synthetic contours")
    

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(exp_mask)
    ax[0].set_title('Experimetnal Image')
    ax[0].axis('off')

    ax[1].imshow(syn_mask)
    ax[1].set_title('Synthetic Image')
    ax[1].axis('off')

    fig.show()

    if debug:
        e_img_m = experimental_image.copy()
        cv.drawContours(e_img_m,[experimental_contour],0,(255,0,0),5)
        s_img_m = synthetic_image.copy()
        cv.drawContours(s_img_m,[synthetic_contour],0,(255,0,0),5)

        return experimental_contour, synthetic_contour, e_img_m,s_img_m
    else:
        return experimental_contour, synthetic_contour
