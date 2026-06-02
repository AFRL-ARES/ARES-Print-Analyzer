from PyAres import Analysis, AnalysisRequest, Outcome
import subprocess
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import re
import pandas as pd

def subprocess_analyzer(request: AnalysisRequest) -> Analysis:
    #inputs
    image_bytes: bytes = request.inputs["Image"]
    experiment_name: str = request.request_metadata.experiment_id
    campaign_name: str = request.request_metadata.campaign_name
    # settings
    model_file: str = request.settings["Model Path"]
    config_json: str = request.settings["Config JSON Path"]
    model_json: str = request.settings["Model JSON Path"]
    output_dir: str = request.settings["Output Path"]
    output_level: int = request.settings["Output Level"]
  
    # 1. Confrim that all necesary files exist and are in the right format
    if (not Path(model_file).exists()) or Path(model_file).suffix != '.stl':
        print("The specifed model file does not exist or is not a .stl file")
        return Analysis(1e5, Outcome.FAILURE)
    if not Path(config_json).exists():
        print("The specifed configuaton JSON file does not exist")
        return Analysis(1e5, Outcome.FAILURE)
    if not Path(model_json).exists():
        print("The specifed model information JSON file does not exist")
        return Analysis(1e5, Outcome.FAILURE)
  
    # 2. Create the output path if it does not exist
    output_path = Path(output_dir) / campaign_name / experiment_name
    output_path.mkdir(exist_ok=True, parents=True)

    # 3. Write the input image to the output path so that the subprocess can access it and so we have it
    image_path = output_path / str(experiment_name + "_base_image.png")
    if output_level >= 0:                               
        with open(str(image_path), "wb") as f:
            f.write(image_bytes)

    # Get the path to your pipeline script
    script_path = str(Path('./src/ares_print_analyzer/contour_similarity_analyzer.py').resolve())

    # 3. Build the command
    # This ensures the subprocess uses the same python environment as our gRPC service
    python_executable = sys.executable
    command = [
        python_executable,
        script_path,
        "--",
        "--image-path", str(image_path),
        "--model-file-path", model_file,
        "--config-json-path", config_json,
        "--model-json-path", model_json,
        "--experiment-name", experiment_name,
        "--output-path", str(output_path),
        "--output-level", str(output_level)
        ]


    # 4. Run the subprocess and capture its output
    try:
        # We run the command and capture stdout/stderr as text
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # This will raise an error if Blender fails
            timeout=240
        )
        
        # 5. Get the score from the script's standard output
        
        # The score and outcome flag are returned in a dict with the keys SCORE and OUTCOME. We parse the string and eval it to get the data.
        regex = re.compile(r'\{[\s\S]*?\}')
        result_string = regex.search(result.stdout).group()
        results_dict = eval(result_string)
        new_stdout = regex.split(result.stdout)[0]
        score = float(results_dict['SCORE'])
        outcome_flag = results_dict['OUTCOME']
        print('--- Analysis Completed ---')
        print(f'\tObjective Score: {score}')
        print('--- Analyzer Output ---')
        print(new_stdout)

        if outcome_flag:
            outcome = Outcome.SUCCESS
        else:
            outcome = Outcome.WARNING # so that ARES OS doesn't die
            print("ANALYSIS FAILURE! CHECK OUTPUT OF ANALYZER!")
            print("##### STDERR #####")
            print(result.stderr)

        update_swapfile(request.settings,score)
        # 6. Return the score in your gRPC response
        return Analysis(result=score, outcome=outcome)

    except subprocess.CalledProcessError as e:
        # Blender script failed
        error_message = f"Blender pipeline failed: {e.stderr}"
        print(error_message, file=sys.stderr)
        return Analysis(result=1e5, error_string=error_message, outcome=Outcome.WARNING)
        
    except Exception as e:
        # Other error (e.g., timeout, can't find blender.exe)
        error_message = f"Internal server error: {e}"
        print(error_message, file=sys.stderr)
        return Analysis(result=1e5, error_string=error_message, outcome=Outcome.WARNING)

def convert_image_bytes_to_ndarray(image_bytes) -> np.ndarray:
  nparr = np.frombuffer(image_bytes, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  if isinstance(img_np, np.ndarray):
    return img_np
  
  else:
    return np.empty(0)
  
def update_swapfile(settings,result):
    # Updates the last row of the swap file with the result
    swap_file = settings.get('Swap File','')
    if swap_file != '' and Path(swap_file).exists() and Path(swap_file).is_file():
        df = pd.read_csv(str(swap_file))
        df.loc[len(df)-1,'objective'] = result
        df.to_csv(str(swap_file),index=False)