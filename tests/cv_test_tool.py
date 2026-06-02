from ares_print_analyzer import pose_and_render
from ares_print_analyzer import contour_analyzer
from pathlib import Path
import cv2

test_image_folder = '/Users/artsloan/Downloads/DOD SAFE-MuugvNcfuZnv2WS7'
problems_folder = '/Users/artsloan/Downloads/DOD SAFE-MuugvNcfuZnv2WS7/problems'
test_image = "tests/test_data/test_images/test_2.png"
model_file = "tests/test_data/bunny_head_0_marked.stl"
config_json = "tests/test_data/config.json"
model_json = "tests/test_data/bunny_head_0_marked.json"
output_dir = "tests/test_data/test_output"

image_files = list(Path(test_image_folder).glob('*.png'))
n_files = len(image_files)

for i, file in enumerate(image_files):
    e_name = file.stem.split('_')[0]
    print(f"Testing File {i+1}/{n_files}: Name: {e_name}")
    img = cv2.imread(str(file))
    output_path = str(Path(output_dir)/e_name)
    try:
        # cv2.imshow('test_image',img)
        # cv2.waitKey(0)
        # c_img, r_img, roi_min, roi_max, obj_center, image_space_contours = pose_and_render(img,model_file,config_json,model_json,output_dir,e_name,output_level=4)
        output_dict = contour_analyzer(str(file),model_file,config_json,model_json,output_path,e_name,output_level=4)
        print(f'Score: {output_dict["SCORE"]}')
    except Exception as e:
        cv2.imwrite(problems_folder+'/'+e_name+'_base_image.png',img)
        print(e)
    


    