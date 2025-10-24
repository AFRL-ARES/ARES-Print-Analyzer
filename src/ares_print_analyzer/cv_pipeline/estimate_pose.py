import cv2 as cv

import numpy as np


def estimate_pose(object_points: np.ndarray,
                  image_points: np.ndarray,
                  config_data: dict,
                  debug: bool = False,
                  image_for_debug: np.ndarray | None = None):
    # 1. configuration_data from JSON 
    try:
        camera_matrix = np.array(config_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(config_data['distortion_coefficients'], dtype=np.float32)
        model_bounds_min =  np.array(config_data['object_bounds_min'], dtype=np.float32)
        model_bounds_max = np.array(config_data['object_bounds_max'], dtype=np.float32)

    except KeyError as e:
        print(f"Error: Missing expected key in configuration data - {e}")
        return None, None
    
    # Perform Pose Estimation using solvePnP
    try:
        dist_coeffs=np.zeros(4) # We've already corrected the distortion, so we're ignoring the coefficeints from the Json.

        success, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            raise Exception("Pose estimation failed")
    except cv.error as e:
        print(f"An OpenCV error occurred in solvePnP: {e}")
        return None, None
    except Exception as e:
        print(f'{e}')
        return None, None

    # Debug Mode: Draw axes and markers on the image
    if debug:
        if image_for_debug is None:
            print("Warning: Debug mode is on, but no image was provided to draw on.")
        else:
            debug_image = image_for_debug.copy()            

            # Draw the bounding box 
            bl, _ = cv.projectPoints(model_bounds_min,rvec, tvec, camera_matrix, dist_coeffs)
            tr, _ = cv.projectPoints(model_bounds_max,rvec, tvec, camera_matrix, dist_coeffs)
            tl, _ = cv.projectPoints(np.array([model_bounds_min[0], model_bounds_max[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
            br, _ = cv.projectPoints(np.array([model_bounds_max[0], model_bounds_min[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
            bl = np.squeeze(bl).astype(int)
            tr = np.squeeze(tr).astype(int)
            tl = np.squeeze(tl).astype(int)
            br = np.squeeze(br).astype(int)

            cv.line(debug_image,tl,tr,(255,0,0),5)
            cv.line(debug_image,tr,br,(255,0,0), 5)
            cv.line(debug_image,br,bl,(255,0,0), 5)
            cv.line(debug_image,bl,tl,(255,0,0), 5)

            # Draw the 3D coordinate axes on the image
            cv.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, 10) # is the length of the axis in mm

            return rvec, tvec, debug_image
    else:
        return rvec, tvec 
