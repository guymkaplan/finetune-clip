import hashlib
import base64
import logging
import re
import random
import time
from functools import wraps

from PIL import Image
import requests
from io import BytesIO
from urllib.parse import urlparse

def retry_exp_backoff(attempts=3, initial_delay=1, backoff_factor=2):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.info(f"Attempt {attempt+1} failed with {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor * random.uniform(1, 1.5)
            raise Exception(f"All {attempts} retry attempts failed")
        return wrapper
    return deco

@retry_exp_backoff()
def load_image(image_source: str, target_size) -> Image.Image:
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
        logging.info(f"Error loading image from {image_source}: {e}")
        raise e
