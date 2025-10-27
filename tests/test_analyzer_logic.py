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
                "Output Path":output_dir}
    req = request(inputs,settings)
    score, success = analyze(req)

    print(f"Success? {success}, Score:{score}")

