from ares_print_analyzer.render_pipeline import render_synthetic_image
import numpy as np
import json
from pathlib import Path
import cv2 as cv

if __name__ == '__main__':
    # %% Configure
    model_file = "/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/Print Analyzer Development/athena-demo-resources/stl files/cv markers/3dbenchy_marked.stl"
    camera_json = "/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/Print Analyzer Development/athena-demo-resources/resources/camera_calibration.json"
    debug_folder = '../athena-demo-resources/debug'
    test_name = 'X150Y105Z110'
    img_W, img_H = 1920,1080

    # %% 
    camera_json = Path(camera_json)
    model_file = Path(model_file)
    debug_folder = Path(debug_folder)
    output_folder = debug_folder / test_name
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(camera_json, mode="r", encoding="utf-8") as read_file:
        config_data = json.load(read_file)
    K = np.array(config_data['camera_matrix'])

    rvec = np.array([[1.95733624],
                     [-1.98630388],
                     [-0.07067832]])
    
    tvec = np.array([[2.04573378],
                     [-0.70774577],
                     [142.04434296]])
    
    filament_color = (255,140,85)
    bed_color = (55,50,50)

    r_img = render_synthetic_image(str(model_file),K,rvec,tvec,filament_color,bed_color,W=img_W,H=img_H,debug=True,debug_folder=str(output_folder))
    cv.imwrite(str(output_folder/(test_name+'_render.jpg')),r_img)
