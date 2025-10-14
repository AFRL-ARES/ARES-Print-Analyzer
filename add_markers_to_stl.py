import numpy as np
from stl import mesh
def add_markers_to_stl(input_path, output_path, padding=5.0, marker_thickness=1.0):
    """
    Adds circular and square markers to an STL file for computer vision pose estimation.

    Places four circular markers on the bottom face (at the corners of the bounding box)
    and a square marker at the midpoint of each of the 12 edges of the bounding box.

    Args:
        input_path (str): Path to the input STL file.
        output_path (str): Path to save the modified STL file.
        padding (float): Distance to add to each side of the object's bounding box.
        marker_thickness (float): The thickness of the generated markers.
    """
    try:
        object_mesh = mesh.Mesh.from_file(input_path)
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return
    
    # Find the bounding box of the object
    bbox_min, bbox_max, bbox_center, bbox_extent = find_bbox(object_mesh)
    # Get rid of the z component for 2D placement of the markers on the print bed
    bbox_min = bbox_min[:2]
    bbox_max = bbox_max[:2]
    bbox_center = bbox_center[:2]
    bbox_extent = bbox_extent[:2]


    # Addd padding to the bounding box
    bbox_min -= padding
    bbox_max += padding

    all_markers = []

    # Circle Markers go on the corners of the bounding box
    corners = [
        np.array([bbox_min[0], bbox_min[1], 0]),
        np.array([bbox_max[0], bbox_min[1], 0]),
        np.array([bbox_max[0], bbox_max[1], 0]),
        np.array([bbox_min[0], bbox_max[1], 0]),
    ]
    for corner in corners:
        circle_marker = make_circular_marker(corner, 8, 0.5)
        all_markers.append(circle_marker)


    # need to find the center of each edege to for the square markers

    # edges proceding clocwise from top face
    edge_midpoints_and_normals = [((bbox_center[0], bbox_max[1]), np.array((0,1))), # top edge
        ((bbox_max[0], bbox_center[1]), np.array((1,0))), # right edge
        ((bbox_center[0], bbox_min[1]), np.array((0,-1))), # bottom edge
        ((bbox_min[0], bbox_center[1]), np.array((-1,0))), # left edge
    ]

    # generate squares

    # Merge and save mesh objects to new STL files
    combined_data = np.concatenate([object_mesh.data] + [m.data for m in all_markers])
    combined_mesh = mesh.Mesh(combined_data)
    combined_mesh.save(output_path)
    print(f"Successfully saved new STL file to '{output_path}'")


#   
#     # --- 2. Generate Circular Markers at Bottom Corners ---
#     # Corners on the Z-min face of the bounding box
#     corners = [
#         np.array([min_coords[0], min_coords[1], min_coords[2]]),
#         np.array([max_coords[0], min_coords[1], min_coords[2]]),
#         np.array([max_coords[0], max_coords[1], min_coords[2]]),
#         np.array([min_coords[0], max_coords[1], min_coords[2]]),
#     ]
    
#     # Normal points down, away from the object
#     circle_normal = np.array([0, 0, -1])

#     for corner in corners:
#         # Offset the marker slightly so it sits on the bounding box face
#         center = corner + circle_normal * marker_thickness / 2.0
#         circle_marker = create_circular_marker(center, marker_size, marker_thickness, circle_normal)
#         all_markers.append(circle_marker)
    
#     print(f"Added 4 circular markers of radius {marker_size:.2f}.")

#     # --- 3. Generate Square Markers at Edge Midpoints ---
#     edge_midpoints_and_normals = [
#         # Bottom face edges
#         ((min_coords + np.array([max_coords[0], min_coords[1], min_coords[2]]))/2, np.array([0, -1, 0])),
#         ((np.array([max_coords[0], min_coords[1], min_coords[2]]) + np.array([max_coords[0], max_coords[1], min_coords[2]]))/2, np.array([1, 0, 0])),
#         ((np.array([max_coords[0], max_coords[1], min_coords[2]]) + np.array([min_coords[0], max_coords[1], min_coords[2]]))/2, np.array([0, 1, 0])),
#         ((np.array([min_coords[0], max_coords[1], min_coords[2]]) + min_coords)/2, np.array([-1, 0, 0])),
#         # Top face edges
#         ((np.array([min_coords[0], min_coords[1], max_coords[2]]) + np.array([max_coords[0], min_coords[1], max_coords[2]]))/2, np.array([0, -1, 0])),
#         ((np.array([max_coords[0], min_coords[1], max_coords[2]]) + max_coords)/2, np.array([1, 0, 0])),
#         ((max_coords + np.array([min_coords[0], max_coords[1], max_coords[2]]))/2, np.array([0, 1, 0])),
#         ((np.array([min_coords[0], max_coords[1], max_coords[2]]) + np.array([min_coords[0], min_coords[1], max_coords[2]]))/2, np.array([-1, 0, 0])),
#         # Vertical edges
#         ((min_coords + np.array([min_coords[0], min_coords[1], max_coords[2]]))/2, np.array([0, -1, 0])),
#         ((np.array([max_coords[0], min_coords[1], min_coords[2]]) + np.array([max_coords[0], min_coords[1], max_coords[2]]))/2, np.array([1, 0, 0])),
#         ((np.array([max_coords[0], max_coords[1], min_coords[2]]) + max_coords)/2, np.array([0, 1, 0])),
#         ((np.array([min_coords[0], max_coords[1], min_coords[2]]) + np.array([min_coords[0], max_coords[1], max_coords[2]]))/2, np.array([-1, 0, 0])),
#     ]

#     for midpoint, normal_guess in edge_midpoints_and_normals:
#         # Refine normal to ensure it points away from the bbox center
#         vec_to_midpoint = midpoint - bbox_center
#         # Find the cardinal axis most aligned with the vector to the midpoint
#         normal = np.zeros(3)
#         normal[np.argmax(np.abs(vec_to_midpoint))] = np.sign(vec_to_midpoint[np.argmax(np.abs(vec_to_midpoint))])
        
#         center = midpoint + normal * marker_thickness / 2.0
#         square_marker = create_square_marker(center, marker_size, marker_thickness, normal)
#         all_markers.append(square_marker)
        
#     print(f"Added 12 square markers of size {marker_size:.2f}.")

#     # --- 4. Combine and Save Meshes ---
#     combined_data = np.concatenate([main_mesh.data] + [m.data for m in all_markers])
#     combined_mesh = mesh.Mesh(combined_data)
#     combined_mesh.save(output_path)
#     print(f"Successfully saved new STL file to '{output_path}'")

def find_bbox(object_mesh):
    '''
    Finds the axis-aligned bounding box of a mesh.
    ''' 

    min_coords = object_mesh.min_
    max_coords = object_mesh.max_
    bbox_center = (min_coords + max_coords) / 2.0   
    bbox_dims = max_coords - min_coords

    return min_coords, max_coords, bbox_center, bbox_dims

def make_circular_marker(center, radius, thickness, segments=32)-> mesh.Mesh:
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
    normal = np.array([0,0,-1], dtype=float)  # Normal pointing downwards
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
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        point_offset = radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
        points_top.append(center + point_offset + thickness * normal)
        points_bottom.append(center + point_offset - 0.0 * normal)

    faces = []
    # Create faces for the top and bottom caps
    for i in range(segments):
        p_top_1 = points_top[i]
        p_top_2 = points_top[(i + 1) % segments]
        p_bottom_1 = points_bottom[i]
        p_bottom_2 = points_bottom[(i + 1) % segments]
        
        center_top = center + thickness * normal
        center_bottom = center - 0.0 * normal

        # Top face
        faces.append([center_top, p_top_1, p_top_2])
        # Bottom face
        faces.append([center_bottom, p_bottom_2, p_bottom_1])
        # Side faces (two triangles per segment)
        faces.append([p_top_1, p_bottom_1, p_top_2])
        faces.append([p_bottom_1, p_bottom_2, p_top_2])

    # Create the mesh
    marker_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
    for j in range(3):
        marker_mesh.vectors[i][j] = points[f[j],:]
    return marker_mesh


if __name__ == "__main__":
    input_stl = "/Users/artsloan/Downloads/3dbenchy.stl"
    output_file = "/Users/artsloan/Downloads/_marked.stl"
    add_markers_to_stl(input_stl, output_file)
    # output_stl = "test_files/CalibrationCube_with_markers.stl"
    # add_markers_to_stl(input_stl, output_stl)