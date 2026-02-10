import bpy
import numpy as np
import cv2 as cv
from mathutils import Matrix, Vector
import tempfile 
from pathlib import Path
def render_synthetic_image(model_file:str,
                           camera_intrinsic:np.ndarray,
                           rvec:np.ndarray,
                           tvec:np.ndarray,
                           filament_rgb:tuple | np.ndarray, 
                           bed_rgb:tuple | np.ndarray,
                           W:int = 1920,
                           H:int = 1080,
                           output_folder: str | None= None,
                           experiment_name : str  = '',
                           output_level :int=0) -> np.ndarray:
    
    bpy.ops.wm.read_homefile() # Reset blender with the default scene to ensure a clean slate for rendering
    init_scene(model_file,W,H) # Intialize the scene with user supplied model    
    cam_cv2blend(camera_intrinsic)
    set_camera_position(rvec,tvec)
    setup_lights()
    set_object_color(filament_rgb)
    set_world_color(bed_rgb)
    make_ground_plane(bed_rgb)
    img = render_image()

    if output_level >=4:
        output_path = Path(output_folder) / (experiment_name+"_scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

    bpy.ops.wm.read_factory_settings()
    return img

def init_scene(model_file,W,H):
    # Blender loads in with a default camera, light, and cube
    # So we need to remove the cube and add in the STL to render
    if "Cube" in bpy.data.objects:
        mesh = bpy.data.objects["Cube"]
        bpy.data.objects.remove(mesh)
    bpy.ops.wm.stl_import(filepath=model_file)
    # Remove the inital light, since we'll emulate the camera light ring later
    if 'Light' in bpy.data.objects:
        light = bpy.data.objects["Light"]
        bpy.data.objects.remove(light)

    # Give the model a material so we can set its color later
    mat = bpy.data.materials.new(name='Model_Mat')
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value = 1.0 # Roughness value of 0.2
    mat.node_tree.nodes["Principled BSDF"].inputs[3].default_value = 5.0 # IOR of 2
    mat.node_tree.nodes["Principled BSDF"].inputs[13].default_value = 0.4 # Specular IOR level of 0.1
    
    bpy.ops.object.material_slot_add()
    bpy.context.object.material_slots[0].material = mat

    bpy.data.materials["Model_Mat"].node_tree.nodes["Principled BSDF"].inputs[2].default_value = 1.0


    # Configure the rendered scene to the approriate size and approximately correct colors
    scene = bpy.context.scene
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'High Contrast'


    # Set the camera view distance
    cam = bpy.data.objects["Camera"].data
    cam.clip_start = 0.1
    cam.clip_end = 1000.0

def cam_cv2blend(K): 
    # Configures the blender camera based on the intrinsic camera matrix from OpenCV
    cam = bpy.data.objects["Camera"].data
    scene = bpy.context.scene
    # assume image is not scaled
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert cam.sensor_fit != 'VERTICAL'
    # Assume that the x pixel aspect ratio is 1.0
    assert scene.render.pixel_aspect_x == 1.0
    
    sensor_width = cam.sensor_width
    # Scene (Render) parameters
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    fx = K[0,0] # X focal length in pixels
    fy = K[1,1] # Y focal length in pixels
    cx = K[0,2] # X Camera Shift in pixels
    cy = K[1,2] # Y Camera Shift in pixels
    # Configure the pixel aspect ratio
    pixel_aspect = fy/fx 
    scene.render.pixel_aspect_y = pixel_aspect
    # Configures the camera focal length
    focal_lenth = (fx * sensor_width)/ w # focal length in mm
    cam.lens = focal_lenth
    # configures the shift of the camera optical center
    cam.shift_x = 0.5 - cx/w
    cam.shift_y = (cy - 0.5 * h) / w

def set_camera_position(rvec,tvec):
    # Get the roation and translation of the camera from the
    # roation and translation vectors from the openCV
    # pose estimation
    cam = bpy.data.objects["Camera"]
    R,_ = cv.Rodrigues(rvec) # world coordinates to CV coordinates
    Rt = R.T # CV coordinates to world coordinates

    cv2blend_r = np.array([[1, 0, 0], 
                         [0, -1, 0], 
                         [0, 0, -1]])
    rot = Rt @ cv2blend_r
    loc = Rt @ -tvec

    rot = Matrix(rot.tolist())
    loc = Vector(loc.flatten())

    cam.matrix_world = Matrix.Translation(loc) @ rot.to_4x4()

def setup_lights():
    # add an array of 12 lights at a dstance of 30 from the camera, all paretned to the camera
    n = 12
    dr = 30
    dz = -5
    cam = bpy.data.objects["Camera"]
    light_data = bpy.data.lights.new(name="LED", type='AREA')
    light_data.shape ='DISK'
    light_data.energy = 9500
    light_data.size=3
    for i in np.arange(n):
        light_name = "Light_{}".format(i)

        # Create new object, pass the light data 
        light_object = bpy.data.objects.new(name=light_name, object_data=light_data)
        # Link object to collection in context
        bpy.context.collection.objects.link(light_object)
        light_object.parent=cam
        # Change light position
        a = 2*np.pi/n
        light_object.location = ((dr/2)*np.cos(i*a), (dr/2)*np.sin(i*a), dz)

def set_object_color(color_rgb):
    # convert the supplied color to HSV, bump up the saturation to compensate for rendering effects
    rgb = np.atleast_1d(color_rgb).astype(np.uint8)
    hsv = cv.cvtColor(rgb.reshape(1,1,3),cv.COLOR_RGB2HSV) 
    hsv[:,:,1] = hsv[:,:,1]
    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2RGB).flatten() 
    vals = rgb/255
    vals = (vals[0],vals[1],vals[2],1)
    bpy.data.materials["Model_Mat"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = vals

def set_world_color(color_rgb):
    rgb = np.atleast_1d(color_rgb).astype(np.uint8)
    hsv = cv.cvtColor(rgb.reshape(1,1,3),cv.COLOR_RGB2HSV) 
    hsv[:,:,1] = hsv[:,:,1]
    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2RGB).flatten() 
    vals = rgb/255
    vals = (vals[0],vals[1],vals[2],1)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = vals

def render_image():
    temp_path = tempfile.gettempdir()
    img_path = Path(temp_path)/"render.png"
    bpy.context.scene.eevee.use_raytracing = True
    bpy.context.scene.eevee.ray_tracing_method = 'PROBE'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    # There is no way to access the render directlty so we write a temporary file and read it back in
    bpy.context.scene.render.filepath = str(img_path)
    bpy.ops.render.render(write_still = True)
    # read the image and then delete the temporary file
    img = cv.imread(img_path)
    img_path.unlink()

    return img

def make_ground_plane(color_rgb):
    rgb = np.atleast_1d(color_rgb).astype(np.uint8)
    hsv = cv.cvtColor(rgb.reshape(1,1,3),cv.COLOR_RGB2HSV) 
    hsv[:,:,1] = hsv[:,:,1]
    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2RGB).flatten() 
    vals = rgb/255
    vals = (vals[0],vals[1],vals[2],1)

    # Define arrays for holding data    
    myvertex = []
    myfaces = []

    # Create all Vertices

    # vertex 0
    mypoint = [(-1000.0, -1000.0, 0.0)]
    myvertex.extend(mypoint)

    # vertex 1
    mypoint = [(1000.0, -1000.0, 0.0)]
    myvertex.extend(mypoint)

    # vertex 2
    mypoint = [(-1000.0, 1000.0, 0.0)]
    myvertex.extend(mypoint)

    # vertex 3
    mypoint = [(1000.0, 1000.0, 0.0)]
    myvertex.extend(mypoint)

    # -------------------------------------
    # Create all Faces
    # -------------------------------------
    myface = [(0, 1, 3, 2)]
    myfaces.extend(myface)
    
    mymesh = bpy.data.meshes.new("Ground Plane")
    myobject = bpy.data.objects.new("Ground Plane", mymesh)
    bpy.context.scene.collection.objects.link(myobject)
    # Generate mesh data
    mymesh.from_pydata(myvertex, [], myfaces)
    # Calculate the edges
    mymesh.update(calc_edges=True)
    # create a material to give the ground plane some color

    mat = bpy.data.materials.new(name='Ground_Mat')
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[1].default_value = 0.0 # Metallic 0.2
    mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value = 0.25 # Roughness value of 0.3
    mat.node_tree.nodes["Principled BSDF"].inputs[3].default_value = 1.1 # IOR of 1.5

    
    # bpy.ops.object.material_slot_add()
    # myobject.material_slots[0].material = mat
    myobject.data.materials.append(mat)
    bpy.data.materials["Ground_Mat"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = vals



