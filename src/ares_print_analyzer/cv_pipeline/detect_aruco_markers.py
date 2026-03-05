import cv2 as cv
import numpy as np

class MarkerDetectionError(RuntimeError):
    """Raised when markers cannot be detected."""
    pass

def detect_aruco_markers(img, debug=False):
    """
    Detects ArUco markers in a given image, handling both normal and inverted grayscale images to account for varying marker/printbed contrast.

    Args:
        img (numpy.ndarray): Input image in BGR format.
        debug (bool, optional): If True, returns a debug image with detected markers drawn. Defaults to False.

    Returns:
        tuple: 
            - corners (tuple of numpy.ndarray): Tuple of arrays containing the corner points of each detected marker.
            - ids (numpy.ndarray): Array of detected marker IDs.
            - debug_image (numpy.ndarray, optional): Image with detected markers drawn (only if debug=True).

    Raises:
        Exception: If no ArUco markers are detected in either the normal or inverted grayscale image.
    """
    inverted = False
    # markers are expected to be black and white, so convert the image to grayscale
    # The lights on the nebula camera tend to show up as blue and can cause glare on the print bed. 
    # Throwing away the blue channel before doing grayscale conversion helps increase the contrast 
    gr_img =np.copy(img)
    gr_img[:,:,0] = 0
    gray = cv.medianBlur(cv.cvtColor(gr_img, cv.COLOR_BGR2GRAY),5)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    equalized = cv.convertScaleAbs(equalized, alpha=1.5, beta=0)
    # equalized = cv.normalize(equalized,None,0,255,cv.NORM_MINMAX)
    gray_inv = (255-gray)
    equalized_inv = (255 - equalized)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters()
    arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementMinAccuracy = 0.01
    arucoParams.cornerRefinementWinSize = 7
    arucoParams.cornerRefinementMaxIterations = 200

    

    # aruco markers are expected to have light pixles on the inside, so try both options
    # this should let the detection work whether the filament is lighter or darker than the
    # printbed

    # Default assumption is that the filament is a lighter shade than the bed
    (corners, ids, _) = cv.aruco.detectMarkers(equalized_inv, aruco_dict,parameters=arucoParams)
    if len(corners) == 0:
        (corners, ids, _) = cv.aruco.detectMarkers(equalized, aruco_dict,parameters=arucoParams)
        if len(corners) > 0:
            inverted = True
    
    
    if len(corners) == 0:
        raise MarkerDetectionError("Could not locate any ArUco markers to set model orientation")

    return corners, ids, inverted
