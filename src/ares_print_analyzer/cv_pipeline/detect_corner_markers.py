import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops_table
from scipy.ndimage import binary_fill_holes

def round_up_odd_int(num):
    """
    Rounds up the input number to the nearest odd integer.

    Parameters
    ----------
    num : float or int
        The number to round up.

    Returns
    -------
    int
        The nearest odd integer greater than or equal to num.
    """
    return int(np.ceil(num) // 2 * 2 + 1)

# find the circular quartered markeers at the corner of the bounding box
def detect_corner_markers(img, inverted=False, debug=False):
    """
    Detects and refines corner marker positions in an image using adaptive thresholding and morphological operations.
    This function processes an image to find circular corner markers, filters out noise and unwanted detections,
    and returns refined corner positions. It is specifically designed to work with markers that appear as circles
    with internal patterns.
    Parameters
    ----------
    img : ndarray
        Input image in BGR color format
    inverted : bool, optional
        If True, inverts the grayscale image before processing. Used when filament is darker than print bed.
        Default is False.
    debug : bool, optional
        If True, saves a debug image showing detected markers and refined corners.
        Default is False.
    Returns
    -------
    ndarray
        Array of refined corner positions with subpixel accuracy. Returns empty array if no corners are detected.
        Shape is (n,2) where n is number of corners detected.
    Notes
    -----
    - Uses adaptive thresholding with parameters scaled based on image dimensions
    - Filters detections based on circularity and area to remove noise
    - Handles cases where text/letters on print bed might be falsely detected as markers
    Examples
    --------
    >>> img = cv.imread('print_bed.jpg')
    >>> corners = detect_corner_markers(img)
    >>> corners = detect_corner_markers(img, inverted=True, debug=True)  # With debug output
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001) # Criteria for corner refinement
    # Grayscale versions of image
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # if the Aruco finding routine had to invert the image to succede 
    if inverted: # for if the filament is darker than the print bed
        gray = (255-gray)

    centers_1 = []
    r_centers_1 = []

    # Segment the image using adaptive thresholding
    # blur and thresholding values work for the nebula camera which has a resoltuion of 1080x1920
    # the parameters are all scaled based on the smaller dimension of the image 
    min_dim = np.min(gray.shape)
    mask_img = cv.adaptiveThreshold(cv.medianBlur(gray,round_up_odd_int(min_dim/128)),
                                    255,
                                    cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY,
                                    round_up_odd_int(min_dim/16),
                                    0
                                    )

    #this should produce a mask with the markers identifed and surrounded by a border but also a lot of 
    # small blobs all over the image due to noise and uneven lighting
    kernel = np.ones((3,3),np.uint8)
    mask_img = cv.morphologyEx(mask_img,cv.MORPH_OPEN,kernel)

    # Right now we're jsut trying to find cirlces, so we can fill any holes that may be in the markers
    mask_uf = mask_img.copy() # an unfilled copy of the mask to use later
    mask_img = binary_fill_holes(mask_img)
    mask_img = (255 * mask_img.astype(np.uint8)) # reconvert to something that opencv likes


    # now we need to filter out miscelanous objects resutling from noise
    # these will tend to either be very small blops or have large fractial extents 
    # sprawling fractial objects will have a large convex hull area relative to their area
    # We'll use the skimages 'area_filled' property since the markers also have open spaces

    l = label(mask_img)
    props = regionprops_table(l,mask_img,['label',
                                    'area',
                                    'perimeter',
                                    'centroid',
                                    'eccentricity',
                                    'equivalent_diameter'])
    # Define a circularity parameter by comapring the equivalent diameter of a region 
    # calcualted from its pixle area to the equivlaent diameter calcualted from its perimiter
    # for a circle this would obviously be 1, for a square it will be ~0.88.
    # Since squares will pick up some corner rounding so set the thresold to about 0.93
    # At this point there should only be the large cirlces that make up the markers and tiny dots
    # so we'll throw away anything smaller than 1000 pixles in area
    props['circularity'] = np.sqrt(4*props['area']/np.pi) / (props['perimeter']/np.pi)
    idx = np.bitwise_and(props['circularity'] >= 0.93, props['area'] > 1000)

    # Sometimes the cirlce finding will pick up letering on the print bed (The letter o)
    # We need to descriminate between the targets and the extra circles so 
    # After we get rid of all the miscleanous junk using the indexing we've already done
    # we can look at the internal areas of each of the identified circles 
    # the things we want to get rid of will have highly circular featrues so we can use the same approch to find them.
    mask_img[np.isin(l,props['label'][~idx])] = 0
    mask_uf[np.isin(l,props['label'][~idx])] = 0
    
    d_mask = np.logical_and(mask_img,~mask_uf)
    d_l = label(d_mask)
    d_props= regionprops_table(d_l,d_mask,['label',
                                        'area',
                                        'perimeter'])
    d_props['circularity'] = np.sqrt(4*d_props['area']/np.pi) / (d_props['perimeter']/np.pi)
    # The things we want to get rid of will have both large circularity and area. 
    # We need the area criteria because the small circle quarters can appear roughly circular if the image processing 
    # left them disconnected in the middle. If they are connected they will have a low circularity
    d_idx = np.bitwise_and(d_props['circularity'] >= 0.93, d_props['area'] > 500 )

    # Now that we have the region we need to get rid of, we need to find its lable in the orignal image
    # and from there get its index in the original props dictt

    d_mask[np.isin(d_l,d_props['label'][~d_idx])] = 0
    bad_labels = np.unique(l[d_mask])
    bad_idx = np.argwhere(np.isin(props['label'],bad_labels))
    idx[bad_idx] = False

    # Get the centroids and diameters of the downslected regions
    centers_1 = np.column_stack((props['centroid-1'][idx],props['centroid-0'][idx])) 
    diameters = props['equivalent_diameter'][idx]

    if np.any(centers_1):
        r_centers_1 = cv.cornerSubPix(gray, np.float32(centers_1), (7,7), (-1,-1), criteria)
    else:
        raise Exception("Could not locate the centers of any corner markers")

    if debug:
        debug_img = img.copy()
        for i, c in enumerate(centers_1):
            cv.circle(debug_img,c.astype(np.int64),int(diameters[i])//2,(0, 255, 0), 3)
            res = np.hstack((centers_1,r_centers_1))
            res = res.astype(np.int32)
            debug_img[res[:,1],res[:,0]]=[0,0,255]
            debug_img[res[:,3],res[:,2]] = [0,255,0]
        return r_centers_1, debug_img
    else:
        return r_centers_1
