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
                    output_level: int = 0,
                    skip_distortion_correction=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    

    # Ensure the output folder(s) exist, redundant with the one created in the main analyzer function but good to have here for debugging purposes when this function is used on its own
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load in config from JSON files
    try:
        with open(str(config_json), 'r') as f:
            c_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find configuration json file - {e}")
        raise e

    try:
        with open(str(model_json), 'r') as f:
            m_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find model json file - {e}")
        raise e
    
    config_data = c_data | m_data

    # section exists to allow the re-analysis of previously distortion corrected images
    if not skip_distortion_correction:
        c_img = correct_distortion(img,config_data)
    else:
        c_img = img
    img_W = c_img.shape[1]
    img_H = c_img.shape[0]
    if output_level >= 0:
        # Save the undistorted image to the output folder
        cv.imwrite(str((output_path / (experiment_name+"_undistorted.jpg"))),c_img)

    try:
        corners, ids, inverted = detect_aruco_markers(c_img)
        circle_centers, circle_diameters = detect_corner_markers(c_img,inverted)
    except Exception as e:
        print(f"An error occured during marker detection: {e}")
        raise e
    
    if output_level >= 3:
        # Save the ArUco debug image to the output folder
        d_img = c_img.copy()
        d_img = cv.aruco.drawDetectedMarkers(d_img, corners, ids)
        cv.imwrite(str((output_path / (experiment_name+"_ArUco.jpg"))),d_img)

        # Save the corner marker debug image to the output folder
        d_img = c_img.copy()
        for i, c in enumerate(circle_centers):
            cv.circle(d_img,c.astype(np.int64),int(circle_diameters[i])//2,(0, 255, 0), 3)
        cv.imwrite(str((output_path / (experiment_name+"_corners.jpg"))),d_img)

    # Make the output from the aruco detection easier to work with for the pose estimation step
    corners = tuple(np.squeeze(c) for c in corners)
    ids = ids.ravel()
    try:
        object_points, image_points = orient_markers((ids, corners), circle_centers, config_data)
    except Exception as e:
        print(f"An error occured during marker orientation: {e}")
        raise e
    
    try:
        rvec, tvec = estimate_pose(object_points, image_points, config_data)
    except Exception as e:
        print(f"An error occured during pose estimation: {e}")
        raise e

    if output_level >=3:
        # Save the pose estimation debug image to the output folder

        d_img = c_img.copy()
        camera_matrix = np.array(config_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32) # We're working from the undisorted image
        model_bounds_min = np.array(config_data['object_bounds_min'], dtype=np.float32)
        model_bounds_max = np.array(config_data['object_bounds_max'], dtype=np.float32)

        # Draw the bounding box 
        bl, _ = cv.projectPoints(model_bounds_min,rvec, tvec, camera_matrix, dist_coeffs)
        tr, _ = cv.projectPoints(model_bounds_max,rvec, tvec, camera_matrix, dist_coeffs)
        tl, _ = cv.projectPoints(np.array([model_bounds_min[0], model_bounds_max[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
        br, _ = cv.projectPoints(np.array([model_bounds_max[0], model_bounds_min[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
        bl = np.squeeze(bl).astype(int)
        tr = np.squeeze(tr).astype(int)
        tl = np.squeeze(tl).astype(int)
        br = np.squeeze(br).astype(int)
        cv.line(d_img,tl,tr,(255,0,0),5)
        cv.line(d_img,tr,br,(255,0,0), 5)
        cv.line(d_img,br,bl,(255,0,0), 5)
        cv.line(d_img,bl,tl,(255,0,0), 5)

        # Draw the 3D coordinate axes on the image
        d_img = cv.drawFrameAxes(d_img, camera_matrix, dist_coeffs, rvec, tvec, 10) # is the length of the axis in mm
        cv.imwrite(str((output_path / (experiment_name+"_posed.jpg"))),d_img)

    config_data['opencv_rotation_vector'] = rvec
    config_data['opencv_translation_vector'] = tvec
    K = np.array(config_data['camera_matrix'])
    
    filament_color = config_data['filament_color_rgb']
    bed_color = config_data['bed_color_rgb']

    # We need some way to filter out the contours of markers later on in case they end up overlapping the object
    # what we'll do here is create the contours each marker border in model space and then transform the points into image coordinates
    # These image space contours can then be used later to substract areas where the model and conour overlap which could give weird scoring results
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
    try:
        r_img = render_synthetic_image(model_path,K,rvec,tvec,filament_color,bed_color,W=img_W,H=img_H,output_level=output_level,
                                    output_folder=str(output_path),
                                    experiment_name=experiment_name)
    except Exception as e:
        print(f"An error occured during rendering: {e}")
        raise e

    if output_level >= 2:                             
        # Save the rendered image to the output folder
        cv.imwrite(str((output_path / (experiment_name+"_render.jpg"))),r_img)

    # using the model bounding box and extent data and the pose estimation, figure out the pixels we need to actually do the analysis
    roi_min, roi_max, obj_center = get_analysis_roi(img_W,img_H,config_data)

    return c_img, r_img, roi_min, roi_max, obj_center, image_space_contours

    
        
