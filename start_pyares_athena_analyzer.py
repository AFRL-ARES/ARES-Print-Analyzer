#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /start_pyares_athena_analyzer
# Project: ARES-Print-Analyzer
# Created Date: Tuesday, February 10th 2026, 9:04:17 am
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

from PyAres import AresAnalyzerService, AresDataType
from ares_print_analyzer import subprocess_analyzer

if __name__ == "__main__":
    """
    Analyzer inputs are:
    1. The experimental image - key: "Image"

    Analyzer Settings are: 
    1. The path to the stl model printed - key: "Model Path"
    2. The path to the configuration json containing details about the experimental configuration - key: "Config JSON Path"
    3. The path to the json file containing details about the 3d model - key: "Model JSON Path"
    4. The parent directory for any outputs - key: "Output Path"
    5. Output Level - key: "Output Level"
        The output level determines the types of data saved to the output directory. 
        The higher the output level, the more data is saved. Levels 0-2 are sufficinet for normal production use, 
        while higher levels are more for de-bugging purposes. The levels are as follows:
        0:  The original image as recieved from the camera. (as <experiment_id>_base_image.jpg)
            A computationally undistorted version of the original image. (as <experiment_id>_undistorted.jpg)  
        1:  All level 0 outputs
            A cropped version of the undistorted image with the detected contour overlaid. (as <experiment_id>_expt_contour.jpg)
            A cropped version of the rendered image with the detected contour overlaid. (as <experiment_id>_synth_contour.jpg)
        2:  All level 1 outputs
            A full version of the rendered image. (as <experiment_id>_render.jpg)
        3:  All level 2 outputs
            A copy of the undistorted image with the detected ArUco markers and their IDs overlaid. (as <experiment_id>_ArUco.jpg)
            A copy of the undistorted image with the circular corner markers overlaid. (as <experiment_id>_corners.jpg)
            A copy of the undistorted image with the result of the pose estimation overlaid. (as <experiment_id>_posed.jpg)
        4:  All level 3 outputs
            A .blend file containing the rendered scene, which can be opened in Blender for debugging purposes. (as <experiment_id>_scene.blend)

    Note: Output images will be saved in <Output Path>/campaign_name/experiment_id/
    """
    name = "PyAres Athena Print Analyzer"
    description = "A PyAres implementation of a contour shape analsyis routine based on the " \
                "work of Graig Ganitano (DOI: 10.1007/s40964-023-00480-1)"
    version = "0.5.0"
    port = 7083

    analyzer = AresAnalyzerService(subprocess_analyzer, name, version, description, use_localhost=True, port=port)
  
    analyzer.add_analysis_parameter("Image", AresDataType.BYTE_ARRAY)

    analyzer.add_setting("Model Path", AresDataType.STRING)
    analyzer.add_setting("Config JSON Path", AresDataType.STRING)
    analyzer.add_setting("Model JSON Path", AresDataType.STRING)
    analyzer.add_setting("Output Path", AresDataType.STRING)
    analyzer.add_setting("Output Level", AresDataType.NUMBER, constraints=[0,1,2,3,4])
    analyzer.start()
