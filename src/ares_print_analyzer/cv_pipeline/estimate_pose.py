import cv2 as cv

import numpy as np


def estimate_pose(object_points: np.ndarray,
                  image_points: np.ndarray,
                  config_data: dict):
    # 1. configuration_data from JSON 
    try:
        camera_matrix = np.array(config_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(config_data['distortion_coefficients'], dtype=np.float32)

    except KeyError as e:
        print(f"Error: Missing expected key in configuration data - {e}")
        return None, None
    
    # 2. Perform Pose Estimation using solvePnP
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

    return rvec, tvec
