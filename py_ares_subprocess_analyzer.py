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

from PyAres import AresAnalyzerService, Analysis, AnalysisRequest, AresDataType, Outcome
import subprocess
import sys
import os
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from pathlib import Path

def Analyze(request: AnalysisRequest) -> Analysis:
  # inputs
  image_bytes: bytes = request.inputs["Image"]
  experiment_name: str = request.request_metadata.experiment_id
  campain_name: str = request.request_metadata.campaign_name
  # settings
  model_file: str = request.settings["Model Path"]
  config_json: str = request.settings["Config JSON Path"]
  model_json: str = request.settings["Model JSON Path"]
  output_dir: str = request.settings["Output Path"]

  # 1. Confrim that all necesary files exist and are in the right format
  if (not Path(model_file).exists()) or Path(model_file).suffix != '.stl':
    print("The specifed model file does not exist or is not a .stl file")
    return Analysis(0.0, Outcome.FAILURE)
  if not Path(config_json).exists():
    print("The specifed configuaton JSON file does not exist")
    return Analysis(0.0, Outcome.FAILURE)
  if not Path(model_json).exists():
    print("The specifed model information JSON file does not exist")
    return Analysis(0.0, Outcome.FAILURE)
  
  # 2. Create the output path if it does not exist
  output_path = Path(output_dir) / campain_name / experiment_name
  output_path.mkdir(exist_ok=True, parents=True)
  image_path = f"{output_path}\\base_image.png"

  with open(image_path, "wb") as f:
      f.write(image_bytes)
  
  # Get the path to your pipeline script
  script_dir = os.path.dirname(os.path.realpath(__file__))
  script_path = os.path.join(script_dir, "experimental_test_pipeline.py")

  # 3. Build the command
  # This ensures the subprocess uses the same python environment as our gRPC service
  python_executable = sys.executable
  command = [
      python_executable,
      script_path,
      "--",
      "--image-path", image_path,
      "--model-file-path", model_file,
      "--config-json-path", config_json,
      "--model-json-path", model_json,
      "--experiment-name", experiment_name 
  ]

  # 4. Run the subprocess and capture its output
  try:
      # We run the command and capture stdout/stderr as text
      result = subprocess.run(
          command,
          capture_output=True,
          text=True,
          check=True,  # This will raise an error if Blender fails
          timeout=240
      )
      
      # 5. Get the score from the script's standard output
      # We assume your script *only* prints the final score
      print(result.stderr)
      print(result.stdout)
      score_location = result.stdout.find("SCORE:")
      score = result.stdout[score_location + 6:]
      print(f"Received a final score of {score}")
      score_str = result.stdout.strip()
      float_score = float(score)

      # 6. Return the score in your gRPC response
      return Analysis(result=float_score, outcome=Outcome.SUCCESS)

  except subprocess.CalledProcessError as e:
      # Blender script failed
      error_message = f"Blender pipeline failed: {e.stderr}"
      print(error_message, file=sys.stderr)
      return Analysis(result=-1.0, error_string=error_message)
      
  except Exception as e:
      # Other error (e.g., timeout, can't find blender.exe)
      error_message = f"Internal server error: {e}"
      print(error_message, file=sys.stderr)
      return Analysis(result=-1.0, error_string=error_message)
    

if __name__ == "__main__":
  print("PyAres Print Analyzer")
  description = "A PyAres implementation of a contour shape analsyis routine based on the " \
  "work of Graig Ganitano (DOI: 10.1007/s40964-023-00480-1)"

  """
  Analyzer inputs are:
  1. The experimental image - key: "Image"

  Analyzer Settings are: 
  1. The path to the stl model printed - key: "Model Path"
  2. The path to the configuration json containing details about the experimental configuration - key: "Config JSON Path"
  3. The path to the json file containing details about the 3d model - key: "Model JSON Path"
  4. The parent directory for any outputs - key: "Output Path"
    Note: Output images will be saved in <Output Path>/campaign_name/experiment_name/
  """

  analyzer = AresAnalyzerService(Analyze, "Print Analyzer", "1.0.0", description)

  analyzer.add_analysis_parameter("Image", AresDataType.BYTE_ARRAY)

  analyzer.add_setting("Model Path", AresDataType.STRING)
  analyzer.add_setting("Config JSON Path", AresDataType.STRING)
  analyzer.add_setting("Model JSON Path", AresDataType.STRING)
  analyzer.add_setting("Output Path", AresDataType.STRING)
  analyzer.start()