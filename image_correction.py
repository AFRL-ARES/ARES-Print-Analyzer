import numpy as np
import cv2 as cv
import json
from skimage.measure import label, regionprops_table
from scipy.ndimage import binary_fill_holes
from collections import defaultdict

# Pipeline for correcting raw images from the camera and doing pose estimation necesary for the synthetic image generation

def correct_camera_distortion(image_data,calibration_file):
    DIM = image_data.shape[:2][::-1]
    with open(calibration_file, mode="r", encoding="utf-8") as read_file:
        cal_data = json.load(read_file)
    K = np.array(cal_data['camera_matrix'])
    D = np.array(cal_data['distortion_coefficients'])
    if cal_data['fisheye']:
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)
    else:
        K_new, _ = cv.getOptimalNewCameraMatrix(K, D,DIM,0,DIM)
        map1, map2 = cv.initUndistortRectifyMap(K, D, np.eye(3), K_new, DIM, cv.CV_16SC2)

    undist_image = cv.remap(image_data, map1, map2, interpolation=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)

    return undist_image

def dectect_aruco_markers(img,debug=False):
    # markers are expected to be black and white, so convert the image to grayscale
    gray = cv.medianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY),5)
    gray_inv = (255-gray)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters()

    # aruco markers are expected to have light pixles on the inside, so try both options
    # this should let the detection work whether the filemnt is lighter or datker than the 
    (corners, ids, _) = cv.aruco.detectMarkers(gray, aruco_dict,parameters=arucoParams)
    if len(corners) == 0:
        (corners, ids, _) = cv.aruco.detectMarkers(gray_inv, aruco_dict,parameters=arucoParams)
    if debug:
        out_img = img.copy()
        out_img = cv.aruco.drawDetectedMarkers(out_img,corners,ids)
        cv.imwrite('gray_img.jpg',gray)
        cv.imwrite('gray_inv_jpg.jpg',gray_inv)
        cv.imwrite('marked_image.jpg',out_img)
        
    corners = tuple(np.squeeze(c) for c in corners)
    ids = ids.ravel()
    return corners, ids

# find the circular quartered markeers at the corner of the bounding box
def detect_corner_markers(img,fil_rgb, bed_rgb, debug=False):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement
    # Grayscale versions of image
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_inv = (255-gray)

    '''
    Method 1:   Use the known colors of the filament and the printbed to0
                threshold the image in HSV color space based on the Hue and 
                Saturation values. Omitting Value helps account for the fact that 
                low layer-count objects may be darker. 
    '''
    
    centers_1 = []
    r_centers_1 = []
    # convert the filament and bed colors from rgb to HSV
    fil_hsv = cv.cvtColor(np.array([[fil_rgb]],dtype=np.uint8),cv.COLOR_RGB2HSV)
    bed_hsv = cv.cvtColor(np.array([[bed_rgb]],dtype=np.uint8),cv.COLOR_RGB2HSV)
    # Adding a little blur helps with noise, espeically scattered light on the printbed
    hsv_img = cv.cvtColor(cv.medianBlur(img, 11), cv.COLOR_BGR2HSV)
    # Segment based on the H & S channels
    # mask_img = cv.inRange(hsv_img[:,:,:2],fil_hsv[:,:,:2],bed_hsv[:,:,:2])

    # Segment the image using adaptive thresholding
    # blur and thresholding values work for the nebula camera which has a resoltuion of 1080x1920
    # the parameters are all scaled based on the smaller dimension of the image 
    scale_param = np.min(gray.shape)
    mask_img = cv.adaptiveThreshold(cv.medianBlur(gray,round_up_odd_int(scale_param/128)),
                                    255,
                                    cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY,
                                    round_up_odd_int(scale_param/16),
                                    0
                                    )

    #this should produce a mask with the markers identifed and surrounded by a border but also a lot of 
    # small blobs all over the image due to noise and uneven lighting
    kernel = np.ones((3,3),np.uint8)

    if mask_img is None or mask_img.size == 0:
        if debug:
            print("Warning: mask_img is None or empty after morphology")
        return np.array([])
    
    mask_img = cv.morphologyEx(mask_img,cv.MORPH_OPEN,kernel)

    # Right now we're jsut trying to find cirlces, so we can fill any holes that may be in the markers
    mask_uf = mask_img.copy() # an unfilled copy of the mask to save for later
    mask_img = binary_fill_holes(mask_img)

    if mask_img is None:
    # If binary_fill_holes fails or returns None, handle it gracefully.
    # We should return an empty array for the corner centers.
        if debug:
            print("Warning: binary_fill_holes returned None or failed.")
        return np.array([])
    
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
    mask_img[np.isin(l,props['label'][~idx])] = 0 # type: ignore
    mask_uf[np.isin(l,props['label'][~idx])] = 0 # type: ignore

    #Adjusted to fix Pylance warnings
    d_mask = np.logical_and(mask_img, ~mask_uf.astype(bool))
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

    d_mask[np.isin(d_l, d_props['label'][~d_idx])] = 0 # type: ignore
    bad_labels = np.unique(l[d_mask]) # type: ignore
    bad_idx = np.argwhere(np.isin(props['label'],bad_labels))
    idx[bad_idx] = False

    # Get the centroids and diameters of the downslected regions
    centers_1 = np.column_stack((props['centroid-1'][idx],props['centroid-0'][idx])) 
    diameters = props['equivalent_diameter'][idx]

    

    if np.any(centers_1):
        r_centers_1 = cv.cornerSubPix(gray, np.float32(centers_1), (15,15), (-1,-1), criteria) # type: ignore

    if debug:
        out_img = img.copy()
        for i, c in enumerate(centers_1):
            cv.circle(out_img,c.astype(np.int64),int(diameters[i])//2,(0, 255, 0), 3)
            res = np.hstack((centers_1,r_centers_1))
            res = res.astype(np.int32)
            out_img[res[:,1],res[:,0]]=[0,0,255]
            out_img[res[:,3],res[:,2]] = [0,255,0]
        cv.imwrite('c_circle_img.jpg',out_img)

    return r_centers_1

def round_up_odd_int(num):
    return int(np.ceil(num) // 2 * 2 + 1)

def estimate_pose(detected_arucos, detected_circles, camera_params_file, model_points_file, image_for_debug=None, debug_mode=False):
    """
    Estimates the pose of a 3D printed object relative to the camera.

    This function calculates the rotation (rvec) and translation (tvec) vectors
    that represent the camera's view of the object. It uses a combination of
    ArUco markers and circular corner markers for robust detection.

    Args:
        detected_arucos (dict): A dictionary where keys are ArUco marker IDs (int)
                                and values are their 4x2 corner coordinates (np.array)
                                in the image plane.
                                Example: {0: np.array([[[...]]]), 2: np.array([[[...]]])}
        detected_circles (np.array): An array of (x, y) coordinates for the centers
                                     of detected circular corner markers.
                                     Example: np.array([[x1, y1], [x2, y2], ...])
        camera_params_file (str): Path to the JSON file containing camera calibration data.
        model_points_file (str): Path to the JSON file containing the 3D coordinates
                                 of the model's reference points.
        image_for_debug (np.array, optional): The original image to draw debug
                                              visualizations on. Required if
                                              debug_mode is True. Defaults to None.
        debug_mode (bool, optional): If True, saves an image with detected points
                                     and coordinate axes drawn. Defaults to False.

    Returns:
        tuple: A tuple containing the rotation vector (rvec) and translation
               vector (tvec). Returns (None, None) if pose estimation fails.
    """
    # 1. Load calibration and model data from JSON files
    try:
        with open(camera_params_file, 'r') as f:
            camera_data = json.load(f)
        camera_matrix = np.array(camera_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(camera_data['distortion_coefficients'], dtype=np.float32)

        with open(model_points_file, 'r') as f:
            model_data = json.load(f)
        # Convert to numpy arrays for easier indexing
        model_aruco_corners = np.array(model_data['aruco_corners'], dtype=np.float32)
        model_circle_centers = np.array(model_data['circle_centers'], dtype=np.float32)
        model_bounds_min =  np.array(model_data['object_bounds_min'], dtype=np.float32)
        model_bounds_max = np.array(model_data['object_bounds_max'], dtype=np.float32)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
        return None, None
    except KeyError as e:
        print(f"Error: Missing expected key in JSON file - {e}")
        return None, None

    # 2. Correlate 2D image points with 3D object points
    object_points = []
    image_points = []

    # Add points from detected ArUco markers
    marker_ids = detected_arucos[0]
    corners = detected_arucos[1]
    aruco_dict = dict(zip(marker_ids,corners))
    aruco_centroids = dict()
    aruco_approx_vectors = dict()
    for id, c in zip(marker_ids, corners):
        # The marker ID directly corresponds to the index in our model file
        if id < len(model_aruco_corners):
            # We use the top-left corner of the ArUco marker
            object_points.append(model_aruco_corners[id])
            image_points.append(c[0]) # Extract the (x,y) of the first corner
            # make some approximate orientation vectors based on the corners of the aruco boxes.
            v1 = c[1]-c[0]
            v1 /= np.linalg.norm(v1) # +x unit vector
            v2 = c[0]-c[3]
            v2 /= np.linalg.norm(v2) # +y unit vector
            aruco_approx_vectors[id]=(v1,v2)
            aruco_centroids[id] = np.mean(c,axis=0)

    # Match detected circle markers to model corners using ArUco markers as references
    if detected_circles is not None and len(detected_circles) > 0:
        circle_dict = dict()
        if len(detected_circles) == 4:
            # If all 4 markers are visible, we only need the direction data from a single aruco
            # which is encoded in the rotational transformation
            # With all 4 corners visible, we can find the midpoint of the corners and make
            # that the origin of the coordinate system

            # grab the first aruco derived unit vectors we have
            orient_vec = list(aruco_approx_vectors.values())[0]
            rot_mat = np.column_stack(orient_vec)
            # transform the corrdiantes of the circles using the vectors from the aruco to put them in approximately the right relative locations
            transformed_circles = np.matvec(rot_mat,detected_circles)

            midppoint = np.mean(transformed_circles,axis=0)
            circle_vecs = np.sign(transformed_circles - midppoint)
            def which_corner(x):
                if np.all(x == [-1.,1.]):
                    id = 0
                elif np.all(x == [1.,1.]):
                    id = 1
                elif np.all(x == [1.,-1.]):
                    id = 2
                elif np.all(x == [-1.,-1.]):
                    id = 3
                return id # type: ignore
            
            for i, r in enumerate(circle_vecs):
                id = which_corner(r)
                circle_dict[id] = detected_circles[i]

        else:
            '''
                Things get a little more tricky for three points since we can't assume orthonginalty 
                or perfect linearity of corners with their neigboring arucos, even with a coordinate rotation

                For this situation, we find the two nearest neighbors to each aruco. Becasuse there is the possibility
                that we might pick up an adjacent and non-adjacent corner depending on the view angle 
                This step include an angle check to make sure the angle between the the points at the 
                aruco is nearly 180 degrees, if it isn't the further point is thrown out

                By comparing the list of nearest neighbors to the known adjacnecy structure of the grid we can determin which points 
                go with which index
            '''
            def get_angle(a,b,o):
                # helper to get the angle between nearest neighbor points given the two points (a and b) and the center point (o)

                # recast coordinates as vectors
                A = a - o
                B = b - o
                mag_A = np.sqrt(A[0]*A[0] + A[1]*A[1])
                mag_B = np.sqrt(B[0]*B[0] + B[1]*B[1])
                angle = np.arccos(np.dot(A,B)/(mag_A*mag_B))

                return angle*180/np.pi # convert to degrees
            
            def get_dist(a,c):
                # helper to compute the distance of each corner point in (c) from the aruco marker centroid (a)
                return np.linalg.vector_norm(c-a,axis=1)

            nearest_neighbors = dict()
            for id, ar in aruco_centroids.items():
                dist = get_dist(ar,detected_circles)
                sort_args = np.argsort(dist)
                nn = sort_args[:2]
                vec_angle = get_angle(detected_circles[nn[0]],detected_circles[nn[1]],ar)
                if vec_angle < 170: 
                    nn = nn[0] # toss out the further point
                    
                nearest_neighbors[id] = np.atleast_1d(nn)

            # invert the nearest neighbor dict
            inverse_nn = defaultdict(list)
            for a_id, c_ids in nearest_neighbors.items():
                for c_id in c_ids:
                        inverse_nn[c_id].append(a_id)
                # try:
                #     for c_id in c_ids:
                #         inverse_nn[c_id].append(a_id)
                # except:
                #     inverse_nn[c_ids].append(a_id)
            
            # refrence adjaceny structre and its inverse
            aruco_to_corner_neighbors = {0: (0,1), 1:(2,1), 2:(3,2), 3:(3,0)} # this is orderd so that the axis value increased from index zero to index 1
            corner_to_aruco_neighbors = defaultdict(list)
            for a_id, c_ids in aruco_to_corner_neighbors.items():
                for c_id in c_ids:
                    corner_to_aruco_neighbors[c_id].append(a_id)
        
            unassigned_circles = np.arange(0,len(detected_circles))
            assignment_dict = dict() 

            # now we can see if any of the corners have two adjenct arucos
            # this positively id's the location
            for c_id, a_ids in inverse_nn.items():
                if len(a_ids) == 2:
                    # find the absolutle corner id from the reference strucure
                    r_0 = aruco_to_corner_neighbors[a_ids[0]]
                    r_1 = aruco_to_corner_neighbors[a_ids[1]]
                    intersect = np.intersect1d(r_0,r_1)[0]

                    if np.isin(c_id,unassigned_circles):
                        circle_dict[intersect] = detected_circles[c_id]
                        unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                        assignment_dict[intersect]=c_id
            if len(circle_dict) == 0:
                # no corners adjacnet to to identified arucos are present
                # need to find another way. Use the same vector approach we used when 4 were present,
                #  but now that we know which ponts are closest to the arucos we can use relative positions as long as we have both corners

                for id, nn in nearest_neighbors.items():

                    rn = aruco_to_corner_neighbors[id]
                    orient_vec = aruco_approx_vectors[id]
                    rot_mat = np.column_stack(orient_vec)

                    # transform the corrdiantes of the circles using the vectors from the aruco to put them in approximately the right relative locations
                    trans_circles = np.matvec(rot_mat,detected_circles[nn])
                    trans_centroid = np.matmul(rot_mat,aruco_centroids[id])

                    trans_vec = trans_circles-trans_centroid
                    if trans_vec.shape[0] > 1:
                        span = np.ptp(trans_vec,axis=0)
                        if span[0] > span[1]: # Alignment along the X axis
                                c_id = nn[np.argmin(trans_vec[:,0])]
                                r_id = rn[0]
                                circle_dict[r_id] = detected_circles[c_id]
                                unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                                assignment_dict[r_id]=c_id

                                c_id = nn[np.argmax(trans_vec[:,0])]
                                r_id = rn[1]
                                circle_dict[r_id] = detected_circles[c_id]
                                unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                                assignment_dict[r_id]=c_id

                                
                        elif span[1] > span[0]: # Alignment along the Y axis
                            c_id = nn[np.argmin(trans_vec[:,0])]
                            r_id = rn[0]
                            circle_dict[r_id] = detected_circles[c_id]
                            unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                            assignment_dict[r_id]=c_id

                            c_id = nn[np.argmax(trans_vec[:,0])]
                            r_id = rn[1]
                            circle_dict[r_id] = detected_circles[c_id]
                            unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                            assignment_dict[r_id]=c_id
                    else:
                        c_id = nn[0]
                        if trans_vec[:,0] > trans_vec[:,1]:
                            r_id=rn[int(np.sign(trans_vec[:,0][0]))]
                            circle_dict[r_id] = detected_circles[c_id]
                            unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                            assignment_dict[r_id]=c_id

                        elif trans_vec[:,1] > trans_vec[:,0]:
                            r_id=rn[int(np.sign(trans_vec[:,1][0]))]
                            circle_dict[r_id] = detected_circles[c_id]
                            unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                            assignment_dict[r_id]=c_id




            else:
                # we've fixed one corner now, so we can use reuse the adjacency structure to identify the other points
                for c in unassigned_circles:
                    if any(inverse_nn[c]):
                        a_id = inverse_nn[c][0]
                        corner_candidates = aruco_to_corner_neighbors[a_id]
                        for corner in corner_candidates:
                            if corner not in assignment_dict:
                                circle_dict[corner] = detected_circles[c]
                                unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles)) # type: ignore
                                assignment_dict[corner]=c

        for c in circle_dict:
            object_points.append(model_circle_centers[c])
            image_points.append(circle_dict[c])
    # Ensure we have enough points for solvePnP
    if len(object_points) < 4:
        print(f"Error: Not enough points to perform pose estimation. Found {len(object_points)}, need at least 4.")
        return None, None

    # Convert lists to NumPy arrays
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    # 3. Perform Pose Estimation using solvePnP
    try:
        dist_coeffs=np.zeros(4) # We've already corrected the distortion, so we're ignoring the coefficeints from the Json.

        success, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            print("Warning: solvePnP was not successful.")
            return None, None
    except cv.error as e:
        print(f"An OpenCV error occurred in solvePnP: {e}")
        return None, None

    # 4. Debug Mode: Draw axes and markers on the image
    if debug_mode:
        if image_for_debug is None:
            print("Warning: Debug mode is on, but no image was provided to draw on.")
        else:
            debug_image = image_for_debug.copy()
            # Draw detected ArUco markers and their IDs
            aruco_corners_list = []
            for c in corners:
                aruco_corners_list.append(c.reshape(1,-1,2))
            aruco_ids_list = np.array(list(ids))
            if len(aruco_corners_list) > 0:
                cv.aruco.drawDetectedMarkers(debug_image, aruco_corners_list, aruco_ids_list)

            # Draw detected circle centers
            for center in detected_circles:
                cv.circle(debug_image, tuple(center.astype(int)), 15, (0, 255, 0), -1)

            

            # Draw the bounding box 
            bl, _ = cv.projectPoints(model_bounds_min,rvec, tvec, camera_matrix, dist_coeffs)
            tr, _ = cv.projectPoints(model_bounds_max,rvec, tvec, camera_matrix, dist_coeffs)
            tl, _ = cv.projectPoints(np.array([model_bounds_min[0], model_bounds_max[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
            br, _ = cv.projectPoints(np.array([model_bounds_max[0], model_bounds_min[1],0]),rvec, tvec, camera_matrix, dist_coeffs)
            bl = np.squeeze(bl).astype(int)
            tr = np.squeeze(tr).astype(int)
            tl = np.squeeze(tl).astype(int)
            br = np.squeeze(br).astype(int)

            cv.line(debug_image,tl,tr,(255,0,0),5) # type: ignore
            cv.line(debug_image,tr,br,(255,0,0), 5) # type: ignore
            cv.line(debug_image,br,bl,(255,0,0), 5) # type: ignore
            cv.line(debug_image,bl,tl,(255,0,0), 5) # type: ignore

            # Draw the 3D coordinate axes on the image
            cv.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, 10) # is the length of the axis in mm

            # Save the debug image
            output_path = "pose_estimation_debug.png"
            cv.imwrite(output_path, debug_image)
            print(f"Debug image saved to {output_path}")


    return rvec, tvec, camera_matrix, dist_coeffs



if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt

    test_image = Path("/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/athena-demo-resources/test images/raw/X175Y130Z110.jpg")
    calibration_file = Path("/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/ARES-Print-Analyzer/resources/camera_calibration.json")
    spatial_json = Path("/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/athena-demo-resources/stl files/cv markers/3dbenchy_marked_spatial.json")
    output_path = test_image.parent / (test_image.stem + '_undist.jpg')
    bed_rgb = (60,50,60)
    fil_rgb = (245,211,122)
    marker_size = 8 

    img = cv.imread(str(test_image))
    c_img = correct_camera_distortion(img,calibration_file)
    cv.imwrite(str(output_path),c_img)


    corners, ids = dectect_aruco_markers(c_img,debug=True)
    circle_centers = detect_corner_markers(c_img,fil_rgb,bed_rgb,debug=True)
    rvec, tvec, K, D = estimate_pose((ids, corners),circle_centers,str(calibration_file),str(spatial_json),image_for_debug=c_img,debug_mode=True)

    p,_ = cv.projectPoints(np.array([0.,0.,0.]), rvec, tvec, K, D)
    print(p)

    Rt,_ = cv.Rodrigues(rvec)
    R = Rt.transpose()
    pos = -R @ tvec
