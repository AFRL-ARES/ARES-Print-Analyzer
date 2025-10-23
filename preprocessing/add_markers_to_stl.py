import numpy as np
from stl import mesh
import cv2 as cv
import json

def add_markers_to_stl(input_path,
                        output_path, 
                        padding=8.0, 
                        marker_thickness=0.4, 
                        marker_size=8.0,
                        marker_start = 1):
    """
    Adds circular and square markers to an STL file for computer vision pose estimation.

    Places four circular markers on the bottom face (at the corners of the bounding box)
    and a square marker at the midpoint of each of the 12 edges of the bounding box.

    Args:
        input_path (pathlib.Path): Path to the input STL file.
        output_path (pathlib.Path): Path to save the modified STL file.
        padding (float): Distance to add to each side of the object's bounding box.
        marker_thickness (float): The thickness of the generated markers.
    """
    try:
        object_mesh = mesh.Mesh.from_file(str(input_path))
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return
    
    # Make the subfolder in the output directory
    output_dir = output_path / input_path.stem
    output_dir.mkdir(parents=True,exist_ok=True)
    # Recenter the Object so the cente of the bounding box is at 0,0
    object_mesh  = recenter_stl(object_mesh)

    # Find the bounding box of the object
    bbox_min, bbox_max, bbox_center, bbox_extent = find_bbox(object_mesh)

    # Get rid of the z component for 2D placement of the markers on the print bed
    bbox_min = bbox_min[:2]
    bbox_max = bbox_max[:2]
    bbox_center = bbox_center[:2]
    bbox_extent = bbox_extent[:2]

    # Add padding to the bounding box
    bbox_min -= padding
    bbox_max += padding

    all_markers = []

    # Circle Markers go on the corners of the bounding box, proceding clockwise from the top-left point
    corners = [
        np.array([bbox_min[0], bbox_max[1], 0]),
        np.array([bbox_max[0], bbox_max[1], 0]),
        np.array([bbox_max[0], bbox_min[1], 0]),
        np.array([bbox_min[0], bbox_min[1], 0])]
    
    for corner in corners:
        circle_marker = make_circular_marker(corner, marker_size/2, marker_thickness)
        all_markers.append(circle_marker)

    # need to find the center of each edege to for the aruco markers

    # edges proceding clocwise from top face
    edge_midpoints_and_normals = [((bbox_center[0], bbox_max[1]), np.array((0,1))), # top edge
        ((bbox_max[0], bbox_center[1]), np.array((1,0))), # right edge
        ((bbox_center[0], bbox_min[1]), np.array((0,-1))), # bottom edge
        ((bbox_min[0], bbox_center[1]), np.array((-1,0))), # left edge
    ]
    aruco_corners = [] 
    # generate aruco markers
    # Using the 4x4 dictionary, which has 50 unique IDs (0-49), since we're only using 4 markers, this is plenty

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_ids = []
    for i, (midpoint, normal_guess) in enumerate(edge_midpoints_and_normals):
        # We use I+1 becasue the id:0 aruco has a disconnected square that can be iffy to print. The set [1,2,3,4] is more robust, which is why it is the default
        aruco_marker,aruco_corner = create_aruco_marker(i+marker_start, aruco_dict, marker_size, marker_thickness, midpoint, invert=False)
        aruco_ids.append(i+marker_start)
        if aruco_marker is not None:
            all_markers.append(aruco_marker)
            aruco_corners.append(aruco_corner)
        else:
            print(f"Warning: ArUco marker {i} could not be created.")

    # Collect the locations of the bounding box reference markers in space relative to the model,
    # This information will be necessary for pose estimation, so we'll save it to a JSON file for later use.
    spatial_dict = {'configuration':{'padding':padding,
                                     'marker_thickness':marker_thickness,
                                     'marker_size':marker_size,
                                     'marker_start':marker_start},
                    'object_name':input_path.stem,
                    'object_center':bbox_center.tolist()+[0],
                    'object_bounds_min':bbox_min.tolist()+[0],
                    'object_bounds_max':bbox_max.tolist()+[0],
                    'object_bounds_extent':bbox_extent.tolist()+[0],
                    'marker_size':marker_size,
                    'aruco_ids':aruco_ids,
                    'aruco_corners':np.array(aruco_corners).tolist(),
                    'circle_centers':np.array(corners).tolist()}


    # Make sure the output folder exists
    stl_output_path = output_dir / (input_path.stem+"_marked.stl")
    # Merge and save mesh objects to new STL files
    combined_data = np.concatenate([object_mesh.data] + [m.data for m in all_markers])
    combined_mesh = mesh.Mesh(combined_data)
    combined_mesh.save(str(stl_output_path)) # type: ignore

    json_path = output_dir / (input_path.stem+"_marked.json")
    with open(str(json_path), 'w') as f:
            json.dump(spatial_dict, f)  
    
    print(f"Successfully saved new STL file to '{output_path}'")

def recenter_stl(mesh):
    # Helper function to align the center of the object's bounding box with the origin in the x and y dirrection
    _,_, bbox_center, _ = find_bbox(mesh)
    translation_vec = -np.append(bbox_center[:2],0)
    mesh.translate(translation_vec)
    mesh.update_min()
    mesh.update_max()
    mesh.update_normals()

    return mesh

def find_bbox(obj_mesh):
    '''
    Finds the axis-aligned bounding box of a mesh.
    ''' 

    min_coords = obj_mesh.min_
    max_coords = obj_mesh.max_
    bbox_center = (min_coords + max_coords) / 2.0   
    bbox_dims = max_coords - min_coords

    return min_coords, max_coords, bbox_center, bbox_dims

def make_circular_marker(center, radius, thickness, segments=32, mode="target")-> mesh.Mesh:
    """
    Creates a 3D circular marker (a flat cylinder) as an stl.Mesh object.

    Args:
        center (np.array): The (x, y, z) center of the circle.
        radius (float): The radius of the circle.
        thickness (float): The thickness of the marker.
        normal (np.array): The vector normal to the circle's face.
        segments (int): The number of segments to approximate the circle.

    Returns:
        stl.Mesh: A mesh object representing the circular marker.
    """
    center = np.append(center, 0) if len(center) == 2 else center
    normal = np.array([0,0,1], dtype=float)  # Normal pointing downwards
    # Create an arbitrary vector `u` not parallel to the normal
    u = np.array([1, 0, 0],dtype=float)
    if np.allclose(np.cross(normal, u), 0):
        u = np.array([0, 1, 0])

    # Create two orthonormal vectors `v1` and `v2` in the plane of the circle
    v1 = np.cross(normal, u)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)

    # Generate points for the two faces of the disk
    points_top = []
    points_bottom = []
    center_top = center + thickness * normal
    center_mid = center + (thickness/2.0) * normal
    center_bottom = center - 0.0 * normal
    faces = []
    if mode == 'filled':
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            point_offset = radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            points_top.append(center + point_offset + thickness * normal)
            points_bottom.append(center + point_offset - 0.0 * normal)
            

        # Create faces for the top and bottom caps
        for i in range(segments):
            p_top_1 = points_top[i]
            p_top_2 = points_top[(i + 1) % segments]
            p_bottom_1 = points_bottom[i]
            p_bottom_2 = points_bottom[(i + 1) % segments]
            
            # Top face
            faces.append([center_top, p_top_1, p_top_2])
            # Bottom face
            faces.append([center_bottom, p_bottom_2, p_bottom_1])
            # Side faces (two triangles per segment)
            faces.append([p_top_1, p_bottom_1, p_top_2])
            faces.append([p_bottom_1, p_bottom_2, p_top_2])
    elif mode  == 'target':
        # Generate a marker featuring a target pattern with two filled quarters and a solid outer ring to eneable corner detection.
        inner_radius = 0.75*radius # solid ring is the outer quarter of the cirlce 
        # We generate segments+1 points to make quarter indexing easier without modulo
        outer_points_top, outer_points_bottom = [], []
        inner_points_top, inner_points_bottom = [], []
        outer_mid_points, inner_mid_points, quad_mid_points = [], [], []
        top_outer_mid_points, bottom_outer_mid_points = [], []

        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            outer_offset = radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            inner_offset = inner_radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            outer_midline_offset = (outer_offset + inner_offset) / 2.0
            inner_midline_offset = inner_offset / 2.0

            outer_points_top.append(center_top + outer_offset)
            outer_points_bottom.append(center_bottom + outer_offset)
            inner_points_top.append(center_top + inner_offset)
            inner_points_bottom.append(center_bottom + inner_offset)

            outer_mid_points.append(center_mid + outer_offset)
            inner_mid_points.append(center_mid + inner_offset)
            inner_mid_points.append(center_mid + inner_midline_offset)
            top_outer_mid_points.append(center_top + outer_midline_offset)
            bottom_outer_mid_points.append(center_bottom + outer_midline_offset)

        segments_per_quarter = segments // 4
        quarter_fill = [True, False, True, False] # fill alternating quarters
        quarter_starts = [0, 2 * segments_per_quarter] # Diagonally opposite quarters
        for i in range(segments):
            q = i // segments_per_quarter

            p_outer_top_1, p_outer_top_2 = outer_points_top[i], outer_points_top[i+1]
            p_inner_top_1, p_inner_top_2 = inner_points_top[i], inner_points_top[i+1]
            p_outer_bottom_1, p_outer_bottom_2 = outer_points_bottom[i], outer_points_bottom[i+1]
            p_inner_bottom_1, p_inner_bottom_2 = inner_points_bottom[i], inner_points_bottom[i+1]

            # Top face of the outer ring
            # First we define the center point of the 4 points making up the outer ring to avoid manifold issues
            center_ot = (p_outer_top_1 + p_outer_top_2 + p_inner_top_1 + p_inner_top_2) / 4.0
            faces.extend([[p_outer_top_1, p_outer_top_2, center_ot],
                          [p_outer_top_2, p_inner_top_2, center_ot],
                          [p_inner_top_2, p_inner_top_1, center_ot],
                          [p_inner_top_1, p_outer_top_1, center_ot]])
            
            # Bottom face of the outer ring
            center_ob = (p_outer_bottom_1 + p_outer_bottom_2 + p_inner_bottom_1 + p_inner_bottom_2) / 4.0
            faces.extend([[p_outer_bottom_2, p_outer_bottom_1, center_ob],
                          [p_inner_bottom_2, p_outer_bottom_2, center_ob],
                          [p_inner_bottom_1, p_inner_bottom_2, center_ob],
                          [p_outer_bottom_1, p_inner_bottom_1, center_ob]])
            # wall of the outer ring
            # Create a center point on the plane defined by the outer ring segment points
            center_ow = (p_outer_top_1 + p_outer_top_2 + p_outer_bottom_1 + p_outer_bottom_2) / 4.0
            faces.extend([[p_outer_top_2, p_outer_top_1, center_ow],
                          [p_outer_bottom_2, p_outer_top_2, center_ow],
                          [p_outer_bottom_1, p_outer_bottom_2, center_ow],
                          [p_outer_top_1, p_outer_bottom_1, center_ow]])

            # depending on which quadrant we're in we eitehr need to fill the quarter or create the side wall of the innner ring
            if quarter_fill[q]:
                # make the top face
                faces.append([p_inner_top_1,p_inner_top_2, center_top])
                # make the bottom face
                faces.append([p_inner_bottom_2,p_inner_bottom_1, center_bottom])
                # if we're at the edge of a filled quarter we need to make the side wall
                if i % segments_per_quarter == 0: # first segment of the quarter
                    center_qw = (p_inner_top_1 + p_inner_bottom_1 + center_top + center_bottom) / 4.0
                    faces.extend([[p_inner_top_1, center_top, center_qw],
                                  [center_top, center_bottom, center_qw],
                                  [center_bottom, p_inner_bottom_1, center_qw],
                                  [p_inner_bottom_1, p_inner_top_1, center_qw]])
                    
                elif (i + 1) % segments_per_quarter == 0:
                    center_qw = (p_inner_top_2 + p_inner_bottom_2 + center_top + center_bottom) / 4.0
                    faces.extend([[center_top, p_inner_top_2, center_qw],
                                  [p_inner_top_2, p_inner_bottom_2, center_qw],
                                  [p_inner_bottom_2, center_bottom, center_qw],
                                  [center_bottom, center_top, center_qw]])
            else:
                # Inner side wall of the ring
                center_iw = (p_inner_top_1 + p_inner_top_2 + p_inner_bottom_1 + p_inner_bottom_2) / 4.0
                faces.extend([[p_inner_top_1, p_inner_top_2, center_iw],
                              [p_inner_top_2, p_inner_bottom_2, center_iw],
                              [p_inner_bottom_2, p_inner_bottom_1, center_iw],
                              [p_inner_bottom_1, p_inner_top_1, center_iw]])

    # Create the mesh
    faces = np.round(np.array(faces), decimals=3)
    marker_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    marker_mesh.vectors = np.array(faces) # type: ignore
    marker_mesh.update_normals() # type: ignore
    
    return marker_mesh # type: ignore

def create_aruco_marker(marker_id, dictionary, physical_size, thickness, center, invert=False) -> mesh.Mesh:
    """
    Creates a 3D ArUco marker as an stl.Mesh object.

    Args:
        marker_id (int): The ID of the ArUco marker to generate.
        dictionary (cv2.aruco.Dictionary): The ArUco dictionary object.
        physical_size (float): The side length of the marker in 3D units.
        thickness (float): The thickness of the marker.
        center (np.array): The (x, y, z) center of the marker.
        normal (np.array): The vector normal to the marker's face.
        marker_resolution (int): The number of pixels for the marker image (influences STL detail).
        invert (bool): If True, the geometry will be created for white pixels instead of black.

    Returns:
        stl.Mesh: A mesh object representing the ArUco marker.
    """
    marker_resolution = 6  
    # 1. Generate the 2D ArUco image from the dictionary
    img = cv.aruco.generateImageMarker(dictionary, marker_id, marker_resolution)

    # 2. Set up the 3D plane for the marker
    center = np.append(center, 0) if len(center) == 2 else center
    normal = np.array([0,0,1], dtype=float)
    normal = normal / np.linalg.norm(normal)
    u = np.array([0, -1, 0],dtype=float)
    if np.allclose(np.cross(normal, u), 0):
        u = np.array([-1, 0, 0])
    v1 = np.cross(normal, u); v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1); v2 /= np.linalg.norm(v2)

    # 3. Convert pixels into 3D cubes
    all_faces = []
    bit_size = physical_size / marker_resolution
    
    # Find the 3D coordinate for the grid's starting corner (top-left)
    start_corner = v1 * (-physical_size / 2.0) + v2 * (physical_size / 2.0)
    pixel_value_to_build = 255 if invert else 0
    # With the selectd aruco duecitonary the marker will always be a 6x6 grid
    # the outer ring will alway be black and the inner 4x4 grid will be the id
    for row in range(marker_resolution):
        for col in range(marker_resolution):
            val = img[row, col]
            if val != pixel_value_to_build:
                continue
            
            # To avoid weird STL manifold issues each face will be composed of four corners and 
            # a central point. This will ensure that no two more than two faces share an edge
            bit_center_local = start_corner + v1 * (col + 0.5) * bit_size - v2 * (row + 0.5) * bit_size
            bit_center_world = center + bit_center_local

            # define the 8 conert vertices of the bit
            half_v1_vec = v1 * (bit_size / 2.0)
            half_v2_vec = v2 * (bit_size / 2.0)
            p_tc = bit_center_world + thickness * normal# top-center
            p_bc = bit_center_world - 0.0 * normal # bottom-center

            p_fc = bit_center_world + half_v2_vec + thickness / 2.0 * normal # front-center
            p_bac = bit_center_world - half_v2_vec + thickness / 2.0 * normal # back-center
            p_lc = bit_center_world - half_v1_vec + thickness / 2.0 * normal # left-center
            p_rc = bit_center_world + half_v1_vec + thickness / 2.0 * normal # right-center

            p_tfl = bit_center_world - half_v1_vec + half_v2_vec + thickness * normal # Top-front-left
            p_tfr = bit_center_world + half_v1_vec + half_v2_vec + thickness * normal # Top-front-right
            p_tbl = bit_center_world - half_v1_vec - half_v2_vec + thickness * normal # Top-back-left
            p_tbr = bit_center_world + half_v1_vec - half_v2_vec + thickness * normal # Top-back-right
            p_bfl = bit_center_world - half_v1_vec + half_v2_vec - 0.0 * normal # Bottom-front-left
            p_bfr = bit_center_world + half_v1_vec + half_v2_vec - 0.0 * normal # Bottom-front-right
            p_bbl = bit_center_world - half_v1_vec - half_v2_vec - 0.0 * normal # Bottom-back-left
            p_bbr = bit_center_world + half_v1_vec - half_v2_vec - 0.0 * normal # Bottom-back-right
            # Top and bottom faces are always exterior
            # when defining faces, use righ-hand rule for normals to make sure they point outwards
            # Top faces
            all_faces.extend([[p_tfr, p_tfl, p_tc],
                            [p_tfl, p_tbl, p_tc],
                            [p_tbl, p_tbr, p_tc],
                            [p_tbr, p_tfr, p_tc],
                            ])
            # Bottom faces
            all_faces.extend([[p_bfl, p_bfr, p_bc],
                            [p_bbl, p_bfl, p_bc],
                            [p_bbr, p_bbl, p_bc],
                            [p_bfr, p_bbr, p_bc],
                            ])
            
            #create edge faces for exterior edges of the marker 
            # Front face (v2 [y+] direction)
            # Face is created if this is the topmost row or the pixel above is zero
            if row == 0 or img[row - 1, col] != pixel_value_to_build:
                all_faces.extend([[p_tfl, p_tfr, p_fc],
                                  [p_tfr, p_bfr, p_fc],
                                  [p_bfr, p_bfl, p_fc],
                                  [p_bfl, p_tfl, p_fc]])
                
            # Back face (-v2 [y-] direction)
            # face is created if this is the bottommost row or the pixel below is zero
            if row == marker_resolution - 1 or img[row + 1, col] != pixel_value_to_build:
                all_faces.extend([[p_tbl, p_bbl, p_bac],
                                  [p_bbl, p_bbr, p_bac],
                                  [p_bbr, p_tbr, p_bac],
                                  [p_tbr, p_tbl, p_bac]])
                
            # Left face (-v1 [x-] direction)
            # face is created if this is the leftmost column or the pixel to the left is zero
            if col == 0 or img[row, col - 1] != pixel_value_to_build:
                all_faces.extend([[p_tbl, p_tfl, p_lc],
                                  [p_tfl, p_bfl, p_lc],
                                  [p_bfl, p_bbl, p_lc],
                                  [p_bbl, p_tbl, p_lc]])
                
            # Right face (v1 [x+] direction)
            # face is created if this is the rightmost column or the pixel to the right is zero
            if col == marker_resolution - 1 or img[row, col + 1] != pixel_value_to_build:
                all_faces.extend([[p_tfr, p_tbr, p_rc],
                                  [p_tbr, p_bbr, p_rc],
                                  [p_bbr, p_bfr, p_rc],
                                  [p_bfr, p_tfr, p_rc]])

    if not all_faces:
        return None # type: ignore
    # We're dealing with FDM printing here, so precision isn't super critical
    # rounding to the nearest 10 nm lets us avoid floating point issues
    all_faces = np.round(np.array(all_faces), decimals=5)
    # 4. Create the final mesh from all the cube faces
    marker_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    marker_mesh.vectors = np.array(all_faces) # type: ignore
    marker_mesh.update_normals() # type: ignore
    return marker_mesh, start_corner+center # type: ignore

if __name__ == "__main__":
    from pathlib import Path
    input_stl = Path("../athena-demo-resources/stl files/orignal/3dbenchy.stl")
    output_file = input_stl.parent.parent / 'test_output'
    add_markers_to_stl(str(input_stl), str(output_file))
