# Athena Contour Analysis Print Analyzer 

This 3D print analyzer is designed for use with [Educational ARES](https://github.com/AFRL-ARES/Educational-ARES). The analysis approach is based on the technique outline in in **A hybrid metaheuristic and computer vision approach to closed-loop calibration of fused deposition modeling 3D printers** by Ganitano et al. (DOI: [10.1007/s40964-023-00480-1](https://doi.org/10.1007/s40964-023-00480-1)) with modifications to make it suitable for use with the Educational ARES project and improve the robustness of the computer vision approach. Briefly, this code automates the grading of 3D prints by comparing an image of the physical object (experimental data) against a synthetic reference rendered from the original 3D model (ground truth).

This project runs as a service using the [PyAres library](https://pypi.org/project/PyAres/), allowing it to integrate with Educational ARES.

## Features
* **Automated Pose Estimation**: Uses ArUco and circular markers to automatically align the 3D model with the camera view, removing the need for a fixed, pre-calibrated camera view and enabling multiple prints without the need to clear the bed bewteen each print. 
* **Synthetic Rendering**: Utilizes Blender (`bpy`) to generate a "perfect" reference image of the part.
* **Contour Analysis**: Extracts and compares the shape of the printed part vs. the model using Chi-square statistics on shape histograms.

## Project Structure
* `start_pyares_athena_analyzer.py`: The main entry point. Starts the PyAres service.
* `src/ares_print_analyzer/`: Main package source.
    * `analyzer_subprocess.py`: Handles request processing and isolates the analysis environment.
    * `contour_similarity_analyzer.py`: The core logic script executed for each analysis job.
    * `cv_pipeline/`: Modules for marker detection, distortion correction, and pose estimation.
    * `analysis/`: Algorithms for contour extraction and statistical scoring.
    * `render_pipeline/`: Blender (`bpy`) scripts for rendering the synthetic reference.

## Installation & Environment Setup

This project requires **Python 3.11**. Due to the dependency on `bpy` (Blender as a Python module) and `opencv`, a dedicated environment is highly recommended.

### Option A: Using Anaconda / Miniconda / Miniforge (Recommended)

1.  **Create the environment:**
    ```bash
    conda create -n ares_analyzer python=3.11
    ```
2.  **Activate the environment:**
    ```bash
    conda activate ares_analyzer
    ```
3.  **Install system dependencies (Optional but recommended for OpenCV/Blender):**
    * On Linux, you may need libraries like `libx11-dev` or `libgl1`.
    * *Example (Ubuntu):* `sudo apt-get install libxi6 libgconf-2-4`
4.  **Install the package:**
    Navigate to the project root directory and run:
    ```bash
    pip install .
    ```
    *Note: This will automatically install dependencies listed in `pyproject.toml`, including `bpy`, `numpy`, `opencv-python`, etc.*

### Option B: Using Python venv

1.  **Ensure you have Python 3.11 installed:**
    Check your version:
    ```bash
    python --version
    ```
2.  **Create the virtual environment:**
    ```bash
    python -m venv ares_venv
    ```
3.  **Activate the environment:**
    * **Windows:**
        ```powershell
        .\ares_venv\Scripts\activate
        ```
    * **Linux/macOS:**
        ```bash
        source ares_venv/bin/activate
        ```
4.  **Install the package:**
    ```bash
    pip install .
    ```

## Configuration

The analyzer requires specific inputs provided via the PyAres service call:

1.  **Image**: The raw byte array of the image taken by the camera.
2.  **Model Path**: Absolute path to the `.stl` file of the printed object.
3.  **Config JSON Path**: Path to a JSON file containing experimental setup details (Camera Matrix, Distortion Coefficients, etc.).
4.  **Model JSON Path**: Path to a JSON file containing model-specific details (Marker locations on the build plate).
5.  **Output Path**: Directory where results will be saved.

## Usage

To start the analyzer service, ensure you have activated the approriate python environment:

```bash
python start_pyares_athena_analyzer.py
```
The service will start on port 7083 (localhost) by default. It waits for PyAres requests containing the required inputs.

## Output Levels

The Output Level setting controls the verbosity of the data saved to disk (for debugging and visualization).

    Level 0: Saves the original and undistorted experimental images.

    Level 1: Adds cropped images showing the detected contours on both experimental and synthetic data.

    Level 2: Adds the full rendered synthetic image.

    Level 3: Adds extensive debug images:

        Detected ArUco markers.

        Detected Corner markers.

        Pose estimation visual (bounding box and axes).

    Level 4: Saves a .blend file. You can open this in Blender to inspect the exact scene setup used for rendering.

## Camera Configuration

The computer vision approach requires the calibration of the intrinsic matrix and distortion coefficients for the camera. Please see the [OpenCV documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) on the topic and `tools/camera_calibration.py`. If you are using the creality nebula camera that is part of the recomended hardware configuration, the fisheye camera model and the chessboard calibration pattern tends to give better results. 

## Calibration models and preparing a configuration file.

We recommend the use of the bunny_head model. The original model as well as a version marked with the appropriate computer vision markers (and the associated .json file) is included in `tests/test_data`. To prepare a model for use see `tools/athena_preprocessor.py`.  

