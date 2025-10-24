from ares_print_analyzer.cv_pipeline.correct_distortion import correct_distortion
from ares_print_analyzer.cv_pipeline.detect_aruco_markers import detect_aruco_markers
from ares_print_analyzer.cv_pipeline.detect_corner_markers import detect_corner_markers
from ares_print_analyzer.cv_pipeline.orient_markers import orient_markers
from ares_print_analyzer.cv_pipeline.estimate_pose import estimate_pose
import cv2 as cv
from pathlib import Path
import json

if __name__ == '__main__':
    #%% Test Configuration
    test_image = "../athena-demo-resources/test images/X150Y105Z110.jpg"
    camera_json = '../athena-demo-resources/resources/camera_calibration.json'
    model_json = '../athena-demo-resources/stl files/cv markers/3dbenchy_marked_spatial.json'
    debug_folder = '../athena-demo-resources/debug'

    #%% Setup all the paths and config data
    test_image = Path(test_image)
    calibration_file = Path(camera_json)
    spatial_file = Path(model_json)
    debug_folder = Path(debug_folder)

    output_folder = debug_folder / test_image.stem
    test_name = test_image.stem
    output_folder.mkdir(exist_ok=True,parents=True)

    try:
        with open(str(calibration_file), 'r') as f:
            camera_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")

    try:
        with open(str(spatial_file), 'r') as f:
            model_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")

    config_data = camera_data | model_data

    #%% Run the test
    img = cv.imread(str(test_image))
    c_img = correct_distortion(img,config_data)

    # Save the undistorted image to the debug folder
    cv.imwrite(str((output_folder / (test_name+"_undistorted.jpg"))),c_img)

    corners, ids, inverted, d_img = detect_aruco_markers(c_img,debug=True)
    # Save the ArUco debug image to the debug folder
    cv.imwrite(str((output_folder / (test_name+"_ArUco.jpg"))),d_img)

    circle_centers,d_img = detect_corner_markers(c_img,inverted,debug=True)
    # Save the ArUco debug image to the debug folder
    cv.imwrite(str((output_folder / (test_name+"_corners.jpg"))),d_img)

    object_points, image_points = orient_markers((ids, corners), circle_centers, config_data)

    rvec, tvec, d_img = estimate_pose(object_points, image_points, config_data,debug=True,image_for_debug=c_img)

    # Save the ArUco debug image to the debug folder
    cv.imwrite(str((output_folder / (test_name+"_posed.jpg"))),d_img)

    print(rvec, tvec)