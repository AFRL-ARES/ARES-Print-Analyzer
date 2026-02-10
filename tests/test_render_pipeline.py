from ares_print_analyzer.render_pipeline import render_synthetic_image
import numpy as np
import json
from pathlib import Path
import cv2 as cv

if __name__ == '__main__':
    # %% Configure
    model_file = "tests/test_data/bunny_head_0_marked.stl"
    camera_json = "tests/test_data/config.json"
    debug_folder = 'tests/test_data/test_output'
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

    rvec = np.array([[ 2.14846834],
                    [-2.11689453],
                    [ 0.26400975]])
    
    tvec = np.array([[-16.55437258],
                    [  6.63789423],
                    [ 70.19520408]])
    
    filament_color = (255,140,85)
    bed_color = (55,50,50)

    r_img = render_synthetic_image(str(model_file),K,rvec,tvec,filament_color,bed_color,W=img_W,H=img_H,output_folder=str(output_folder), experiment_name=test_name, output_level=4)
    cv.imwrite(str(output_folder/(test_name+'_render.jpg')),r_img)
