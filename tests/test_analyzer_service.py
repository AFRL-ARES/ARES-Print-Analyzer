from PyAres.test_tools import AnalyzerTestClient
from pathlib import Path
import cv2

def read_image_as_bytes(image_path: Path) -> bytes:
    """Reads an image from the given path and returns its byte representation to emulate the data stream the analyzer service expects.

    Args:
        image_path (Path): Path to the image file.
    """

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    _, buffer = cv2.imencode('.png', img) # type: ignore
    image_bytes = buffer.tobytes()

    return image_bytes

if __name__ == "__main__":
    test_image = "tests/test_data/test_images/test_2.png"
    model_file = "tests/test_data/bunny_head_0_marked.stl"
    config_json = "tests/test_data/config.json"
    model_json = "tests/test_data/bunny_head_0_marked.json"
    output_dir = "tests/test_data/test_output"

    test_image = Path(test_image)
    model_file = Path(model_file)
    model_json = Path(model_json)
    config_json = Path(config_json)
    output_dir = Path(output_dir)

    settings_dict = {
        "Model Path": str(model_file),
        "Config JSON Path": str(config_json),
        "Model JSON Path": str(model_json),
        "Output Path": str(output_dir),
        "Output Level": 4
    }

    img_bytes = read_image_as_bytes(str(test_image))
    request_dict = {"Image": img_bytes}

    test_client = AnalyzerTestClient(port=7083, host='localhost')
    test_client.get_info()
    test_client.check_status()

    test_client.run_analysis(request_dict, settings=settings_dict)
    