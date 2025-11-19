import sys
import argparse
import numpy as np
import cv2
from src.ares_print_analyzer import pose_and_render
from src.ares_print_analyzer.analysis import *
from scipy.optimize import linear_sum_assignment



def convert_image_bytes_to_ndarray(image_bytes) -> np.ndarray:
  nparr = np.frombuffer(image_bytes, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  if isinstance(img_np, np.ndarray):
    return img_np
  
  else:
    return np.empty(0)


def get_script_arguments():
    """
    Parses arguments passed *after* the '--' separator.
    """
    argv = sys.argv
    
    # Get all arguments after '--'
    if "--" not in argv:
        script_args = []
    else:
        script_args = argv[argv.index("--") + 1:]

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Blender Analysis Pipeline")
    parser.add_argument("--image-path",
                        type=str,
                        required=True,
                        help="Path to the image for analysis")
    parser.add_argument("--model-file-path",
                        type=str,
                        required=True,
                        help="Path to the stl model")
    parser.add_argument("--config-json-path",
                        type=str,
                        required=True,
                        help="Path to the config json file")
    parser.add_argument("--model-json-path",
                        type=str,
                        required=True,
                        help="Path to the model json file")
    parser.add_argument("--experiment-name",
                        type=str,
                        required=True,
                        help="Name of the current experiment")
    parser.add_argument("--output-path",
                        type=str,
                        default=".",
                        help="The output path for files created by Blender")
    
    # Parse the arguments
    args = parser.parse_args(script_args)
    return args

def run_analysis_pipeline(image_path, output_path, model_path, config_json_path, model_json_path, experiment_name):    
    # Use stderr for logging, so it doesn't pollute stdout
    print(f"Starting pipeline for: {image_path}", file=sys.stderr)
    try:
      with open(image_path, "rb") as f:
          image_bytes = f.read()
    except FileNotFoundError:
       print(f"Error: Image file not found at {image_path}", file=sys.stderr)
       return 0.0
    except Exception as e:
       print(f"An error occured while loading the image: {e}", file=sys.stderr)
       return 0.0
    
    input_image = convert_image_bytes_to_ndarray(image_bytes)

    # 3. Perform pose estimation on experimental image and render syntheic image
    try:
      experiment_image, synthetic_image, roi_min, roi_max, obj_center = pose_and_render(input_image,
                                                                                        model_path,
                                                                                        config_json_path,
                                                                                        model_json_path,
                                                                                        str(output_path),
                                                                                        experiment_name)
    except Exception as e:
      print("An error occured in the compuer vision pipeline: {}".format(e), file=sys.stderr)
      return 0.0
    
    # 4. Crop the images down to only the ROI and Get the contours from the experimental and synthetic images
    exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    local_center = obj_center - roi_min
    try:
      exp_contour, syn_contour = get_contours(exp_crop,syn_crop,local_center)

    except Exception as e:
      print("An error occured during contour extration: {}".format(e), file=sys.stderr)
      return 0.0
    # 5. Get the histograms for both contours
    try: 
      exp_hist = get_histogram(exp_contour)
      syn_hist = get_histogram(syn_contour)
    except Exception as e: 
      print("An error occured during histogram calculation: {}".format(e), file=sys.stderr)
      return 0.0

    # 6. Score the contours on how similar they are
    try:
      stats = get_chi_statistic(syn_hist,exp_hist)
      row_ind, col_ind = linear_sum_assignment(stats)
      score = stats[row_ind, col_ind].sum()/len(row_ind)

    except Exception as e:
      print("An error occured during scoring: {}".format(e), file=sys.stderr)
      return 0.0
    
    return score

# --- Main execution ---
if __name__ == "__main__":
    
    args = get_script_arguments()
    
    final_score = run_analysis_pipeline(args.image_path, 
                                        args.output_path, 
                                        args.model_file_path, 
                                        args.config_json_path, 
                                        args.model_json_path, 
                                        args.experiment_name)
    
    # CRITICAL: Print *only* the final score to stdout.
    # The gRPC service's subprocess.run() will read this value.
    print(f"FINAL SCORE:{final_score}")