import numpy as np
import cv2 as cv
import json

def correct_distortion(image_data,cal_data):
    """
    Corrects camera lens distortion in an image using calibration parameters.

    This function removes radial and tangential lens distortion from images using
    either standard or fisheye camera models. It uses calibration parameters stored
    in a JSON file to perform the correction.

    Args:
        image_data (numpy.ndarray): Input image as a numpy array.
        calibration_file (str): Path to JSON file containing camera calibration parameters.
            The JSON file should contain:
            - 'camera_matrix': 3x3 camera intrinsic matrix
            - 'distortion_coefficients': Vector of distortion coefficients
            - 'fisheye': Boolean indicating if fisheye model should be used

    Returns:
        numpy.ndarray: Undistorted image as a numpy array with the same dimensions
            as the input image.

    Example:
        >>> img = cv2.imread('distorted_image.jpg')
        >>> undistorted = correct_camera_distortion(img, 'calibration.json')
    """
    DIM = image_data.shape[:2][::-1]
    K = np.array(cal_data['camera_matrix'])
    D = np.array(cal_data['distortion_coefficients'])
    if cal_data['fisheye']:
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)
    else:
        K_new, _ = cv.getOptimalNewCameraMatrix(K, D,DIM,0,DIM)
        map1, map2 = cv.initUndistortRectifyMap(K, D, np.eye(3), K_new, DIM, cv.CV_16SC2)

    undist_image = cv.remap(image_data, map1, map2, interpolation=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)

    return undist_image