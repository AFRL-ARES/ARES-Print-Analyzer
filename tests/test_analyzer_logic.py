from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from pathlib import Path
from ares_print_analyzer import pose_and_render
from ares_print_analyzer.analysis import *

class request(object):
  def __init__(self,inputs,settings):
    self.inputs = inputs
    self.settings = settings

def analyze(request: request) -> tuple[float, bool]:
  # inputs
  input_image: np.ndarray = request.inputs["Image"]
  experiment_name: str = request.inputs["Experiment Name"]
  campain_name: str = request.inputs["Campaign Name"]
  # settings
  model_file: str = request.settings["Model Path"]
  config_json: str = request.settings["Config JSON Path"]
  model_json: str = request.settings["Model JSON Path"]
  output_dir: str = request.settings["Output Path"]
  make_video: bool = request.settings["Make Video"]

  # 1. Confrim that all necesary files exist and are in the right format
  if (not Path(model_file).exists()) | (Path(model_file).suffix != '.stl'):
    raise FileExistsError("The specifed model file does not exist or is not a .stl file")
    return np.inf, False
  if not Path(config_json).exists():
    raise FileExistsError("The specifed configuaton JSON file does not exist")
    return np.inf, False
  if not Path(model_json).exists():
    raise FileExistsError("The specifed model information JSON file does not exist")
    return np.inf, False
  
  # 2. Create the output path if it does not exist
  output_path = Path(output_dir) / campain_name / experiment_name
  output_path.mkdir(exist_ok=True, parents=True)
  # 3. Perform pose estimation on experimental image and render syntheic image
  try:
    experiment_image, synthetic_image, roi_min, roi_max, obj_center = pose_and_render(input_image,
                                                                                      model_file,
                                                                                      config_json,
                                                                                      model_json,
                                                                                      str(output_path),
                                                                                      experiment_name,
                                                                                      debug=True)
  except Exception as e:
    print("An error occured in the compuer vision pipeline: {}".format(e))
    return np.inf, False

  # 4. Crop the images down to only the ROI and Get the contours from the experimental and synthetic images
  exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
  syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
  local_center = obj_center - roi_min
  try:
    exp_contour, syn_contour ,e,s = get_contours(exp_crop,syn_crop,local_center,debug=True)
    cv2.imwrite(str(output_path/"ex_cont.jpg"),e)
    cv2.imwrite(str(output_path/"sy_cont.jpg"),s)

  except Exception as e:
    print("An error occured during contour extration: {}".format(e))
    return np.inf, False
  # 5. Get the histograms for both contours
  try: 
    exp_hist = get_histogram(exp_contour)
    syn_hist = get_histogram(syn_contour)
  except Exception as e: 
    print("An error occured during histogram calculation: {}".format(e))
    return np.inf, False

  # 6. Score the contours on how similar they are
  try:
    stats = get_chi_statistic(syn_hist,exp_hist)
    row_ind, col_ind = linear_sum_assignment(stats)
    score = stats[row_ind, col_ind].sum()/len(row_ind)
  except Exception as e:
    print("An error occured during scoring: {}".format(e))
    return np.inf, False
  #D_MAX = 2.0
  #D_clipped = min(score, D_MAX)
  #normalized_score = 10 * max(1.0 - (D_clipped / D_MAX))
  if make_video:
    from ares_print_analyzer.results_visualization import plot_summary_histogram, plot_polar_representation, plot_histogram_colormap, save_video

    # Create the polar plot and histogram images for both experimental and syntheic histogram
    """
    For both the experimental and syntheic image Generate :
    1. The image
    2. the image with the contour drawn on it (desaturate background)
    3. Overlay the polar plot and resultant histogram for 5 random points for each image
    4. overlay the result histogram over both in the center of the image
    """
    w = obj_center[0]+1920//4*np.array([-1,1])
    if np.max(w) >= 1920:
      diff = np.max(w) -1919
      w -= diff
    elif np.min(w) <=0:
      w += np.min(w)
    rng = np.random.default_rng()
    exp_points = rng.integers(low=0,high=exp_hist.shape[0],size=5)
    syn_points = rng.integers(low=0,high=syn_hist.shape[0],size=5)

    def generate_frames(image,w,points,contour,histogram):
      frames = []
      # adjust the contour to the crop using w
      # Crop the image width, tyring to keep the object as centered as possible
      out_img = image[:,w[0]:w[1],:]
      frames.append(out_img)
      # make a desaturated version of the image to make the contours pop
      cont_img = out_img.copy()
      cont_img = cv2.cvtColor(cont_img,cv2.COLOR_BGR2HSV)
      cont_img[:,:,1] = cont_img[:,:,1] // (1/0.5)
      cont_img = cv2.cvtColor(cont_img,cv2.COLOR_HSV2BGR)
      # we need to adjust the contor to the crop 
      cv2.drawContours(cont_img,[contour-np.array([w.min(),0])],0,(0,255,0),10)
      frames.append(cont_img)
      for p in points:
        working_img = cont_img.copy()
        p_img = plot_polar_representation(contour,p)
       
        _, h_img = plot_histogram_colormap(histogram,p)
        p_img = cv2.resize(p_img,(320,320),interpolation=cv2.INTER_LINEAR)
        h_img = cv2.resize(h_img,(320,320),interpolation=cv2.INTER_LINEAR)
        # insert the polar plot in the top left corner and the histogram on the left right under the first image
        p_rows,p_cols,_ = p_img.shape
        h_rows,h_cols,_ = h_img.shape
        working_img[0:p_rows, 0:p_cols,:] = p_img
        working_img[p_rows:p_rows+h_rows,0:h_cols,:] = h_img
        frames.append(working_img)

      return frames

    e_frames = generate_frames(experiment_image,
                              w,
                              exp_points,
                              exp_contour+roi_min, # contour needs to be adjusted to total image reference frame
                              exp_hist)
    s_frames = generate_frames(synthetic_image,
                              w,
                              syn_points,
                              syn_contour+roi_min, # contour needs to be adjusted to total image reference frame
                              syn_hist)
  
    #combine experimental and synthetic frames into one image
    combined_frames= []
    for ef,sf in zip(e_frames,s_frames):
      new_frame = np.zeros((1080,1920,3),dtype=np.uint8)
      rows,cols,_ = ef.shape

      new_frame[0:rows,0:cols,:] = ef
      new_frame[0:rows,cols:2*cols] = sf

      combined_frames.append(new_frame)

    end_frame = combined_frames[1].copy()

    hist_img = plot_summary_histogram(syn_hist,exp_hist,row_ind,col_ind)
    hist_img = cv2.resize(hist_img,(780,780),interpolation=cv2.INTER_LINEAR)
    h_r,h_c,_ = hist_img.shape
    
    row_range = [1080//2-h_r//2,1080//2+h_r//2]
    col_range = [1920//2-h_c//2,1920//2+h_c//2] 
    end_frame[row_range[0]:row_range[1],col_range[0]:col_range[1],:] = hist_img
    combined_frames.append(end_frame)
    save_video(str(output_path/"demo_video.m4v"),combined_frames)

  return score, True

if __name__ == "__main__":
  image_source_folder = "../athena-demo-resources/test images/raw"
  model_file = "../athena-demo-resources/stl files/cv markers/3dbenchy_marked.stl"
  config_json = "../athena-demo-resources/processed files/config.json"
  model_json = "../athena-demo-resources/stl files/cv markers/3dbenchy_marked_spatial.json"
  output_dir = "../test_output"
#   model_file = str(Path(model_file).resolve()

  campaign_name = "test campaign"
  image_source_folder = Path(image_source_folder)
  for image_file in image_source_folder.glob('*.jpg'):
    expereriment_name = image_file.stem
    img = cv2.imread(str(image_file))
    inputs = {"Image":img,
              'Campaign Name':campaign_name,
              'Experiment Name':expereriment_name}
    settings = {"Model Path": model_file,
                "Config JSON Path":config_json,
                "Model JSON Path":model_json,
                "Output Path":output_dir,
                "Make Video":True}
          
    req = request(inputs,settings)
    score, success = analyze(req)

    print(f"Success? {success}, Score:{score}")

