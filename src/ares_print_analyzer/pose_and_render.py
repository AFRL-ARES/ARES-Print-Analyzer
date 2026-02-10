#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/pose_and_render.py
# Project: ARES-Print-Analyzer
# Created Date: Thursday, October 23rd 2025, 11:17:42 am
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
from .cv_pipeline import *
from .render_pipeline import *
import numpy as np
import cv2 as cv
import json
from pathlib import Path
from itertools import product
import sys

def get_analysis_roi(img_w,img_h,config_data):
    # Get the bounding box plus 10% for good measure
    # then enfoce a square aspect ratio

    bbox_min = np.array(config_data['object_bounds_min'])
    bbox_max = np.array(config_data['object_bounds_max'])
    bbox_height = np.array(config_data['object_bounds_extent'][2])

    K = np.array(config_data['camera_matrix'])
    D = np.zeros(4)
    tvec = np.array(config_data['opencv_translation_vector'])
    rvec = np.array(config_data['opencv_rotation_vector'])
    x_pts = [bbox_min[0],bbox_max[0]]
    y_pts = [bbox_min[1],bbox_max[1]]
    z_pts = [0,bbox_height]

    points = np.array(list(product(*[x_pts,y_pts,z_pts])))
    p,_ = cv.projectPoints(points, rvec, tvec, K, D)
    p = p.astype(np.int32).squeeze()
    p[p<=0] = 0
    p[p[:,0] >= img_w] = img_w
    p[p[:,1] >= img_h] = img_h
    roi_min = np.min(p,axis=0)
    roi_max = np.max(p,axis=0)
    roi_cent = np.mean(np.column_stack((roi_min,roi_max)),axis=1).astype(int)
    roi_span = np.max(np.ptp(np.column_stack((roi_min,roi_max)),axis=1)).astype(int)
    roi_min = roi_cent 
    roi_min = roi_cent - roi_span//2
    roi_max = roi_cent + roi_span//2
    obj_center,_ = cv.projectPoints(np.array([[0.0,0.0,0.0]]), rvec, tvec, K, D)

    p = p.astype(np.int32).squeeze()
    obj_center = obj_center.astype(np.int32).squeeze()
    return roi_min,roi_max, obj_center



def pose_and_render(img: np.ndarray,
                    model_path: str,
                    config_json: str,
                    model_json: str,
                    output_folder: str,
                    experiment_name: str,
                    debug: bool = True,
                    skip_distortion_correction=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Ensure the output folder(s) exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if debug:
        debug_folder = output_folder / 'debug'
        debug_folder.mkdir(parents=True, exist_ok=True)
    
    # Load in config from JSON files
    try:
        with open(str(config_json), 'r') as f:
            c_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}", file=sys.stderr)

    try:
        with open(str(model_json), 'r') as f:
            m_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}", file=sys.stderr)
    
    config_data = c_data | m_data
    if not skip_distortion_correction:
        c_img = correct_distortion(img,config_data)
    else:
        c_img = img
    img_W = c_img.shape[1]
    img_H = c_img.shape[0]
    # Save the undistorted image to the output folder
    cv.imwrite(str((output_folder / (experiment_name+"_undistorted.jpg"))),c_img)

    if not debug:
        corners, ids, inverted = detect_aruco_markers(c_img)
        circle_centers = detect_corner_markers(c_img,inverted)
        
    else:
        corners, ids, inverted, d_img = detect_aruco_markers(c_img,debug=True)
        # Save the ArUco debug image to the debug folder
        cv.imwrite(str((debug_folder / ("debug_"+experiment_name+"_ArUco.jpg"))),d_img)

        circle_centers,d_img = detect_corner_markers(c_img,inverted,debug=True)
        # Save the corner marker debug image to the debug folder
        cv.imwrite(str((debug_folder / ("debug_"+experiment_name+"_corners.jpg"))),d_img)


    object_points, image_points = orient_markers((ids, corners), circle_centers, config_data)

    if debug:
        rvec, tvec, d_img = estimate_pose(object_points, image_points, config_data,debug=True,image_for_debug=c_img)
        # Save the pose_estimation debug image to the debug folder
        cv.imwrite(str((debug_folder / ("debug_"+experiment_name+"_posed.jpg"))),d_img)
    else:
        rvec, tvec = estimate_pose(object_points, image_points, config_data)

    config_data['opencv_rotation_vector'] = rvec
    config_data['opencv_translation_vector'] = tvec
    K = np.array(config_data['camera_matrix'])
    
    filament_color = config_data['filament_color_rgb']
    bed_color = config_data['bed_color_rgb']

    # We need some way to filter out the contorus of markers later on in case they end up overlapping the object
    # what we'll do here is create the contours each marker border in model space and then transform the points into image coordinates
    # These image space controus can then be used later to substract areas where the model an conour overlap which could give weird dcoaring results
    model_space_contours = []
    model_marker_size = config_data['configuration']['marker_size'] # aruco edge length or circle diameter
    model_circle_centers = np.array(config_data['circle_centers'], dtype=np.float32)
    model_aruco_corners = np.array(config_data['aruco_corners'], dtype=np.float32)
    # Circle marker contours approximated by 32 points
    radius = model_marker_size/2
    angles = 2*np.pi*np.arange(32)/32
    points = radius*np.column_stack((np.cos(angles),np.sin(angles),np.zeros_like(angles)))
    for c in model_circle_centers:
        c_points = points+c
        model_space_contours.append(c_points)
    # aruco corner from config file is top left, so we generate the other three and then find the center
    corner_points = np.array([[0,0,0],
                              [model_marker_size,0.0,0],
                              [model_marker_size,-model_marker_size,0],
                              [0.0,-model_marker_size,0]])
    for c in model_aruco_corners:
        c_points = c+corner_points
        model_space_contours.append(c_points)

    image_space_contours = []
    for c in model_space_contours:
        c_2d,_ = cv.projectPoints(c,rvec,tvec,K,np.zeros(4))
        image_space_contours.append(c_2d.astype(int))
    image_space_contours = tuple(image_space_contours)

    if debug:
        r_img = render_synthetic_image(model_path,K,rvec,tvec,filament_color,bed_color,W=img_W,H=img_H,debug=True,debug_folder=str(debug_folder))
    else:
        r_img = render_synthetic_image(model_path,K,rvec,tvec,filament_color,bed_color,W=img_W,H=img_H)
    # Save the Rendered image to the output folder
    cv.imwrite(str((output_folder / (experiment_name+"_render.jpg"))),r_img)

    # using the model bouding box and extent data and the pose estimation, figure out the pixels we need to actually do the analysis
    roi_min, roi_max, obj_center = get_analysis_roi(img_W,img_H,config_data)

    return c_img, r_img, roi_min, roi_max, obj_center, image_space_contours

    
        
