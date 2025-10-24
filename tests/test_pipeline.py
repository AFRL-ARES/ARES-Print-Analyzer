from ares_print_analyzer import pose_and_render
from pathlib import Path
import numpy as np
import cv2 as cv


image_file = "../athena-demo-resources/test images/X150Y105Z110.jpg"
model_file = "/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/Print Analyzer Development/athena-demo-resources/stl files/cv markers/3dbenchy_marked.stl"
config_json = "/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/Print Analyzer Development/athena-demo-resources/processed files/config.json"
model_json = "/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/Print Analyzer Development/athena-demo-resources/stl files/cv markers/3dbenchy_marked_spatial.json"
debug_folder = '../athena-demo-resources/debug'
test_name = 'X150Y105Z110'

e_img = cv.imread(image_file)

c, r, roi_min,roi_max, obj_center = pose_and_render(e_img,
                                                    model_file,
                                                    config_json,
                                                    model_json,
                                                    debug_folder,
                                                    test_name,
                                                    debug=True)

debug_folder = Path(debug_folder)
cv.imwrite(str(debug_folder / 'cam.jpg'),c)
cv.imwrite(str(debug_folder / 'render.jpg'),r)

c_c = c[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
cv.imwrite(str(debug_folder / 'cam_crop.jpg'),c_c)


r_c=r[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
cv.imwrite(str(debug_folder / 'render_crop.jpg'),r_c)
print(roi_min)
print(roi_max)
print(obj_center)
