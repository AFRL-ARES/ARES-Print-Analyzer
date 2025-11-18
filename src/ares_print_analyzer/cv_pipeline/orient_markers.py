import numpy as np
from collections import defaultdict

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

def orient_markers(detected_arucos, detected_circles, config_data):
    # 1. Load calibration and model data from JSON files
    try:
        # Convert to numpy arrays for easier indexing
        model_aruco_corners = np.array(config_data['aruco_corners'], dtype=np.float32)
        model_aruco_ids = np.array(config_data['aruco_ids'], dtype=int)
        model_circle_centers = np.array(config_data['circle_centers'], dtype=np.float32)
        #marker_thickness = config_data['marker_thickness']
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
            # While not the first choice of orientation method, they can be used in a pinch to 
            # transform the image coordinates into something more parsable
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
            # Will all 4 corners we can find the approximate center of the ROI and use signs of the relative locations 
            # of the corners to figure out what goes where.
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
                return id
            
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
            
            # refrence adjaceny structre and its inverse
            # Because the ArUcos are always oriented so that that smallest number is at the top and 
            # values increase clockwise we can support an arbitrary starting ID number
            aruco_to_corner_neighbors = {model_aruco_ids[0]:(0,1),
                                         model_aruco_ids[1]:(2,1),
                                         model_aruco_ids[2]:(3,2),
                                         model_aruco_ids[3]:(3,0)} # this is orderd so that the axis value increased from index zero to index 1
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
                                unassigned_circles = np.delete(unassigned_circles,np.argwhere(c_id==unassigned_circles))
                                assignment_dict[corner]=c

        for c in circle_dict:
            object_points.append(model_circle_centers[c])
            image_points.append(circle_dict[c])

    # Ensure we have enough points for solvePnP
    if len(object_points) < 4:
        raise Exception("Not enough points to perform pose estimation. Found {len(object_points)}, need at least 4.")
    
    # Convert lists to NumPy arrays
    object_points = np.array(object_points, dtype=np.float32)
    # object_points += np.array([0,0,marker_thickness]) # Account for the fact that we'll be looking at the tops of the markers
    image_points = np.array(image_points, dtype=np.float32)

    return object_points, image_points