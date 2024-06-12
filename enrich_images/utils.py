from PIL import Image
import requests
from io import BytesIO
from urllib.parse import urlparse

def load_image(image_source: str, target_size=(128,128)) -> Image.Image:
    """
    Load an image from a URL or local file path.

    Args:
        image_source (str): The URL or local file path of the image.

    Returns:
        PIL.Image.Image: The loaded image in RGB format.

    Raises:
        Exception: If there's an error loading the image.
    """
    try:
        parsed_url = urlparse(image_source)
        if parsed_url.scheme in ("http", "https"):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"Error loading image from {image_source}: {e}")
        raise e
