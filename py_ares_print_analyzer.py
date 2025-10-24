#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /py_ares_print_analyzer.py
# Project: ARES-Print-Analyzer
# Created Date: Tuesday, October 14th 2025, 11:14:37 am
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

from PyAres import AresAnalyzerService, Analysis, AnalysisRequest, AresDataType
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from pathlib import Path
from ares_print_analyzer import pose_and_render
from ares_print_analyzer.analysis import *

def convert_image_bytes_to_ndarray(image_bytes) -> np.ndarray:
  nparr = np.frombuffer(image_bytes, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img_np

def analyze(request: AnalysisRequest) -> Analysis:
  # inputs
  image_bytes: bytes = request.inputs["Image"]
  input_image = convert_image_bytes_to_ndarray(image_bytes)
  experiment_name: str = request.inputs["Experiment Name"]
  campain_name: str = request.inputs["Campaign Name"]
  # settings
  model_file: str = request.settings["Model Path"]
  config_json: str = request.settings["Config JSON Path"]
  model_json: str = request.settings["Model JSON Path"]
  output_dir: str = request.settings["Output Path"]

  # 1. Confrim that all necesary files exist and are in the right format
  if (not Path(model_file).exists()) | (Path(model_file).suffix == '.stl'):
    raise FileExistsError("The specifed model file does not exist or is not a .stl file")
    return Analysis(0.0, False)
  if not Path(config_json).exists():
    raise FileExistsError("The specifed configuaton JSON file does not exist")
    return Analysis(0.0, False)
  if not Path(model_json).exists():
    raise FileExistsError("The specifed model information JSON file does not exist")
    return Analysis(0.0, False)
  
  # 2. Create the output path if it does not exist
  output_path = Path(output_dir) / campain_name / experiment_name
  output_path.mkdir(exist_ok=True, parents=True)
  # 3. Perform pose estimation on experimental image and render syntheic image
  try:
    experiment_image, synthetic_image, roi_min, roi_max, obj_center = pose_and_render(input_image,
                                                                                      model_file,
                                                                                      config_json,
                                                                                      model_json,
                                                                                      str(output_path),
                                                                                      experiment_name)
  except Exception as e:
    print("An error occured in the compuer vision pipeline: {}".format(e))
    return Analysis(0.0, False)

  # 4. Crop the images down to only the ROI and Get the contours from the experimental and synthetic images
  exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
  syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
  local_center = obj_center - roi_min
  try:
    exp_contour, syn_contour = get_contours(exp_crop,syn_crop,local_center)
  except Exception as e:
    print("An error occured during contour extration: {}".format(e))
    return Analysis(0.0, False)
  # 5. Get the histograms for both contours
  try: 
    exp_hist = get_histogram(exp_contour)
    syn_hist = get_histogram(syn_contour)
  except Exception as e: 
    print("An error occured during histogram calculation: {}".format(e))
    return Analysis(0.0, False)

  # 6. Score the contours on how similar they are
  try:
    stats = get_chi_statistic(syn_hist,exp_hist)
    row_ind, col_ind = linear_sum_assignment(stats)
    score = stats[row_ind, col_ind].sum()/len(row_ind)
  except Exception as e:
    print("An error occured during scoring: {}".format(e))
    return Analysis(0.0, False)
  #D_MAX = 2.0
  #D_clipped = min(score, D_MAX)
  #normalized_score = 10 * max(1.0 - (D_clipped / D_MAX))
  
  analysis = Analysis(score, True)
  return analysis

if __name__ == "__main__":
  print("PyAres Print Analyzer")
  description = "A PyAres implementation of a contour shape analsyis routine based on the " \
  "work of Graig Ganitano (DOI: 10.1007/s40964-023-00480-1)"

  """
  Analyzer inputs are:
  1. The experimental image - key: "Image",
  2. The name of the experiment - key: "Experiment Name"
  3. The name of the campaign - key: "Campaign Name"

  Analyzer Settings are: 
  1. The path to the stl model printed - key: "Model Path"
  2. The path to the configuration json containing details about the experimental configuration - key: "Config JSON Path"
  3. The path to the json file containing details about the 3d model - key: "Model JSON Path"
  4. The parent directory for any outputs - key: "Output Path"
    Note: Output images will be saved in <Output Path>/campaign_name/experiment_name/

  """
  

  analyzer = AresAnalyzerService(analyze, "Print Analyzer", "1.0.0", description)

  analyzer.add_analysis_parameter("Image", AresDataType.BYTE_ARRAY)
  analyzer.add_analysis_parameter("Experiment Name", AresDataType.STRING)
  analyzer.add_analysis_parameter("Campaign Name", AresDataType.STRING)

  analyzer.add_setting("Model Path", AresDataType.STRING)
  analyzer.add_setting("Config JSON Path", AresDataType.STRING)
  analyzer.add_setting("Model JSON Path", AresDataType.STRING)
  analyzer.add_setting("Output Path", AresDataType.STRING)
  analyzer.start()