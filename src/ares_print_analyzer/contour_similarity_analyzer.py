#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/contour_similarity_analyzer.py
# Project: ARES-Print-Analyzer
# Created Date: Tuesday, February 10th 2026, 9:40:53 am
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
from PyAres import Analysis, AnalysisRequest, Outcome
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .analysis import get_contours, get_histogram, get_chi_statistic
from .pose_and_render import pose_and_render

def contour_analyzer(request: AnalysisRequest) -> Analysis:
    #inputs
    image_bytes: bytes = request.inputs["Image"]
    experiment_name: str = request.request_metadata.experiment_id
    campaign_name: str = request.request_metadata.campaign_name
    # settings
    model_file: str = request.settings["Model Path"]
    config_json: str = request.settings["Config JSON Path"]
    model_json: str = request.settings["Model JSON Path"]
    output_dir: str = request.settings["Output Path"]
    output_level: int = request.settings["Output Level"]
  
    # 1. Confrim that all necesary files exist and are in the right format
    if (not Path(model_file).exists()) or Path(model_file).suffix != '.stl':
        print("The specifed model file does not exist or is not a .stl file")
        return Analysis(1e5, Outcome.FAILURE)
    if not Path(config_json).exists():
        print("The specifed configuaton JSON file does not exist")
        return Analysis(1e5, Outcome.FAILURE)
    if not Path(model_json).exists():
        print("The specifed model information JSON file does not exist")
        return Analysis(1e5, Outcome.FAILURE)
  
    # 2. Create the output path if it does not exist
    output_path = Path(output_dir) / campaign_name / experiment_name
    output_path.mkdir(exist_ok=True, parents=True)
    image_path = output_path / str(experiment_name + "_base_image.png")
    if output_level >= 0:                               
        with open(str(image_path), "wb") as f:
            f.write(image_bytes)

    # 3. Convert the image bytes to a numpy array for use with the analyzer
    try:
        img = convert_image_bytes_to_ndarray(image_bytes)
    except Exception as e:
        print(f"An error occured while converting the image byte array to a numpy array for analysis: {e}")
        return Analysis(1e5, Outcome.FAILURE)
    
    # 4. Do the pose estimation and rendering steps
    try:
        (experiment_image, 
        synthetic_image, 
        roi_min, roi_max, 
        obj_center, 
        marker_contours) = pose_and_render(img,model_file,config_json,model_json,
                                           str(output_path),experiment_name,output_level=output_level)
    except Exception as e:
        print("An error occured in the compuer vision pipeline: {}".format(e))
        return Analysis(1e5, Outcome.FAILURE)
    #5. Crop the images down to only the ROI and Get the contours from the experimental and synthetic images
    exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    local_center = obj_center - roi_min

    marker_contours = tuple([c - roi_min for c in marker_contours])

    try:
      exp_contour, syn_contour = get_contours(exp_crop,syn_crop,local_center, marker_contours)

    except Exception as e:
      print("An error occured during contour extration: {}".format(e))
      return Analysis(1e5, Outcome.FAILURE)
    
    if output_level >= 1:
       d_img = exp_crop.copy()
       cv2.drawContours(d_img,[exp_contour],0,(0,0,0),7)
       cv2.imwrite(str(output_path/(experiment_name+"_expt_contour.jpg")),d_img)

       d_img = exp_crop.copy()
       cv2.drawContours(d_img,[syn_contour],0,(0,0,0),7)
       cv2.imwrite(str(output_path/(experiment_name+"_synth_contour.jpg")),d_img)

    # 5. Get the histograms for both contours
    try: 
      exp_hist = get_histogram(exp_contour)
      syn_hist = get_histogram(syn_contour)

    except Exception as e: 
      print("An error occured during histogram calculation: {}".format(e))
      return Analysis(1e5, Outcome.FAILURE)


    # 6. Score the contours on how similar they are
    try:
      stats = get_chi_statistic(syn_hist,exp_hist)
      row_ind, col_ind = linear_sum_assignment(stats)
      score = stats[row_ind, col_ind].sum()/len(row_ind)

    except Exception as e:
      print("An error occured during scoring: {}".format(e))
      return Analysis(1e5, Outcome.FAILURE)

    return Analysis(score, Outcome.SUCCESS)



def convert_image_bytes_to_ndarray(image_bytes) -> np.ndarray:
  nparr = np.frombuffer(image_bytes, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  if isinstance(img_np, np.ndarray):
    return img_np
  
  else:
    return np.empty(0)


