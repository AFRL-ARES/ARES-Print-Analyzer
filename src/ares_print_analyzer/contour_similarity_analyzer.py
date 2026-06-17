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
from PyAres import AnalysisResponse, AnalysisRequest, Outcome
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import argparse
# Probably a better way to do this, but this allows the process to be called both a a subprocess and as a standalone script
try:
    from .analysis import get_contours, get_histogram, get_chi_statistic
    from .pose_and_render import pose_and_render
except:
  from ares_print_analyzer.analysis import get_contours, get_histogram, get_chi_statistic
  from ares_print_analyzer.pose_and_render import pose_and_render


def get_script_arguments():
    """
    Parses arguments passed *after* the '--' separator.
    """
    argv = sys.argv
    
    # Get all arguments after '--'
    if "--" not in argv:
        script_args = []
    else:
        script_args = argv[argv.index("--") + 1:]

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Blender Analysis Pipeline")
    parser.add_argument("--image-path",
                        type=str,
                        required=True,
                        help="Path to the image for analysis")
    parser.add_argument("--model-file-path",
                        type=str,
                        required=True,
                        help="Path to the stl model")
    parser.add_argument("--config-json-path",
                        type=str,
                        required=True,
                        help="Path to the config json file")
    parser.add_argument("--model-json-path",
                        type=str,
                        required=True,
                        help="Path to the model json file")
    parser.add_argument("--experiment-name",
                        type=str,
                        required=True,
                        help="Name of the current experiment")
    parser.add_argument("--output-path",
                        type=str,
                        default=".",
                        help="The output path for files created by Blender")
    parser.add_argument("--output-level",
                        type=str,
                        default="0.0",
                        help="The desired level of output detail")
    
    # Parse the arguments
    args = parser.parse_args(script_args)
    return args

def contour_analyzer(image_path:str, model_file:str, config_json:str, model_json:str, output_folder:str, experiment_name:str, output_level:int):
    output_path = Path(output_folder)
    # 1. Read the image file
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image file: {image_path}")
            return {'SCORE': 1e5, 'OUTCOME': False}
    except Exception as e:
        print(f"An error occurred while reading the image file: {e}")
        return {'SCORE': 1e5, 'OUTCOME': False}
    
    # 4. Do the pose estimation and rendering steps
    try:
        (experiment_image, 
        synthetic_image, 
        roi_min, roi_max, 
        obj_center, 
        marker_contours) = pose_and_render(img,model_file,config_json,model_json,
                                           str(output_path),experiment_name,output_level=output_level)
    except Exception as e:
        print("An error occurred in the compuer vision pipeline: {}".format(e))
        return {'SCORE': 1e5, 'OUTCOME': False}
    #5. Crop the images down to only the ROI and Get the contours from the experimental and synthetic images
    exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    local_center = obj_center - roi_min

    marker_contours = tuple([c - roi_min for c in marker_contours])

    try:
      exp_contour, syn_contour = get_contours(exp_crop,syn_crop,local_center, marker_contours)

    except Exception as e:
      print("An error occurred during contour extration: {}".format(e))
      return {'SCORE': 1e5, 'OUTCOME': False}
    
    if output_level >= 1:
       d_img = exp_crop.copy()
       cv2.drawContours(d_img,[exp_contour],0,(0,0,0),7)
       cv2.imwrite(str(output_path/(experiment_name+"_expt_contour.jpg")),d_img)

       d_img = syn_crop.copy()
       cv2.drawContours(d_img,[syn_contour],0,(0,0,0),7)
       cv2.imwrite(str(output_path/(experiment_name+"_synth_contour.jpg")),d_img)

    # 5. Get the histograms for both contours
    try: 
      exp_hist = get_histogram(exp_contour)
      syn_hist = get_histogram(syn_contour)

    except Exception as e: 
      print("An error occurred during histogram calculation: {}".format(e))
      return {'SCORE': 1e5, 'OUTCOME': False}


    # 6. Score the contours on how similar they are
    try:
      stats = get_chi_statistic(syn_hist,exp_hist)
      row_ind, col_ind = linear_sum_assignment(stats)
      score = stats[row_ind, col_ind].sum()/len(row_ind)

    except Exception as e:
      print("An error occurred during scoring: {}".format(e))
      return {'SCORE': 1e5, 'OUTCOME': False}

    return {'SCORE': score, 'OUTCOME': True}

# --- Main execution ---
if __name__ == "__main__":
    
    args = get_script_arguments()
    
    output_dict = contour_analyzer(args.image_path, 
                                    args.model_file_path, 
                                    args.config_json_path, 
                                    args.model_json_path,
                                    args.output_path,  
                                    args.experiment_name,
                                    int(float(args.output_level)))
    
    # CRITICAL: Print *only* the final score to stdout.
    # The gRPC service's subprocess.run() will read this value.
    print(output_dict)