#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/analysis/get_contours.py
# Project: ARES-Print-Analyzer
# Created Date: Friday, October 24th 2025, 10:48:11 am
# Author(s): Graig Gantiano, Nick Kleiner, Arthur W. N. Sloan
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

def morphological_contour_cleanup(input_contour, marker_contours, img_shape, 
                                  erosion_size=3, dilation_size=5, 
                                  min_area_threshold=50):
    """
    Cleans an object contour by subtracting predicted marker regions based on 
    morphological intersection checks.

    Args:
        input_contour: The main object contour (numpy array).
        marker_contours: A tuple/list of predicted marker contours (numpy arrays).
        img_shape: Tuple (height, width) of the image.
        erosion_size: Kernel size for eroding marker mask (intersection check).
        dilation_size: Kernel size for dilating marker mask (subtraction padding).
        min_area_threshold: Minimum area to keep disjoint regions.

    Returns:
        The largest remaining contour after subtraction and cleanup.
    """
    
    # 1. Initialize Master Mask with the input contour filled
    # This represents the object we want to clean
    h, w = img_shape[:2]
    master_mask = np.zeros((h, w), dtype=np.uint8)
    cv.drawContours(master_mask, [input_contour], -1, 255, thickness=cv.FILLED)
    
    # Define Morphological Kernels
    # Ellipse shapes are generally smoother for natural contours than Rects
    erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size))
    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilation_size, dilation_size))

    # 2. Iterate through predicted marker contours
    for marker_cnt in marker_contours:
        
        # A. Create a temporary mask for this specific marker
        marker_mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(marker_mask, [marker_cnt], -1, 255, thickness=cv.FILLED)
        
        # B. Erode the marker mask to create the "Strict Check" region
        # We shrink the marker. If this shrunken region still hits the object,
        # it's definitely a valid overlap, not just edge noise.
        eroded_marker_mask = cv.erode(marker_mask, erode_kernel, iterations=1)
        
        # C. Check Intersection
        # logical AND between the main object and the eroded marker
        intersection = cv.bitwise_and(master_mask, eroded_marker_mask)
        
        if cv.countNonZero(intersection) > 0:
            # D. If Intersection Found: Prepare the Subtraction
            # We go back to the ORIGINAL marker mask and Dilate it (pad it)
            # to ensure we remove the marker traces completely.
            dilated_marker_mask = cv.dilate(marker_mask, dilate_kernel, iterations=1)
            
            # E. Subtract from Master Mask
            # We draw the dilated marker region as Black (0) onto the Master Mask
            # This cuts the hole.
            cv.drawContours(master_mask, [marker_cnt], -1, 0, thickness=cv.FILLED) 
            # Note: To apply the full dilation padding, we subtract using the mask:
            master_mask[dilated_marker_mask > 0] = 0

    # 3. Final Cleanup and Hole Filling
    # RETR_EXTERNAL only retrieves the outer boundary. 
    # If the subtraction created a hole inside the object, this flag ignores it, 
    # effectively "filling" the object back up instantly.
    new_contours, _ = cv.findContours(master_mask, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    
    if not new_contours:
        return None

    # 4. Remove small disconnected regions
    valid_contours = []
    for cnt in new_contours:
        if cv.contourArea(cnt) > min_area_threshold:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    # Return the largest contour (assumption: the object is the largest thing)
    largest_contour = max(valid_contours, key=cv.contourArea)
    
    return largest_contour

def get_contours(experimental_image: np.ndarray,
                 synthetic_image: np.ndarray,
                 object_center: np.ndarray,
                 marker_contours: tuple,
                 debug: bool = False) -> tuple[np.ndarray,np.ndarray]:
    # Returns the opencv contours of the experimental and synthetic object
    img_W = experimental_image.shape[1]
    img_H = experimental_image.shape[0]

    # Convert images to grayscale
    syn_gray = cv.medianBlur(cv.cvtColor(synthetic_image,cv.COLOR_BGR2GRAY), 5)
    exp_gray = cv.cvtColor(experimental_image,cv.COLOR_BGR2GRAY)

    # Use Otsu's method to estimate the right global threshold value and then tweak it a bit to be more greedy so we don't miss details like stringing
    syn_ret , _ = cv.threshold(syn_gray, 0, 255, cv.THRESH_OTSU)
    exp_ret, _ = cv.threshold(exp_gray, 0, 255, cv.THRESH_OTSU)

    _ , syn_mask = cv.threshold(syn_gray, syn_ret*0.9, 255, cv.THRESH_BINARY)
    _ , exp_mask = cv.threshold(exp_gray, exp_ret*0.9, 255, cv.THRESH_BINARY)

    # Find the contours of all white (True) blobs in the mask images.
    # This can return multiple contours, but thanks to the CV pose estimation we know where the 
    # object should be in image coordinates

    exp_cont,_ = cv.findContours(exp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(exp_cont) > 0:
        experimental_contour = downselect_contours(exp_cont, object_center)
    else:
        raise Warning("Could Not find any experimental contours")

    syn_cont,_ = cv.findContours(syn_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(syn_cont) > 0:
        synthetic_contour = downselect_contours(syn_cont,object_center)
    else:
        raise Warning("Could Not find any synthetic contours")
    
    #cleanup any areas where the contours overlap the markers
    experimental_contour = morphological_contour_cleanup(experimental_contour,
                                                         marker_contours,
                                                         experimental_image.shape,
                                                         erosion_size=5,
                                                         dilation_size=10,
                                                         min_area_threshold=50)
    synthetic_contour = morphological_contour_cleanup(synthetic_contour,
                                                         marker_contours,
                                                         experimental_image.shape,
                                                         erosion_size=5,
                                                         dilation_size=10,
                                                         min_area_threshold=50)

    
    return experimental_contour, synthetic_contour