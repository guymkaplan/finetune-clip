import asyncio
import io
import time

import torch
from PIL import Image
from torchvision.transforms import transforms
from transformers import VisionTextDualEncoderModel, HfArgumentParser, \
    AutoImageProcessor
import logging
import os
import pandas as pd

from config.constants import CONSTANTS
from utils import Transform
from utils.utils import load_image

logger = logging.getLogger(__name__)
IMAGE_BYTES_COL_NAME = "_image_bytes_"
def model_fn(model_dir,device):
    """
    Load the model from the directory where SageMaker has downloaded the model artifacts.
    """
    logging.info("starting to load model")
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    state = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
    model = VisionTextDualEncoderModel.from_pretrained(model_dir)
    model.load_state_dict(state, strict=False)
    image_transformations = Transform(
        model.config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std
    )
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info(f"Loaded model to gpu")
    return model, image_transformations


def input_fn(request_body, request_content_type):
    """
    The input_fn pre-processes input data.
    Specifically, the input_fn function is responsible for deserializing your input data so that it can be passed to your model.
    It takes input data and content type as parameters, and returns deserialized data.
    This implementation deserialises a batch of assets from a parquet file.
    """

    assert request_content_type in [
        "application/parquet"
    ], f"Input has unsupported ContentType: {request_content_type}"
    start_time = time.process_time()
    dataset = pd.DataFrame()
    request_bytes = io.BytesIO(request_body)
    if request_bytes.getbuffer().nbytes > 0:
        dataset = pd.read_parquet(request_bytes)
        dataset.columns = dataset.columns.str.lower()
        if dataset.empty is False:
            start_time = time.process_time()
            logger.info("Starting assets download...")
            loop = asyncio.get_event_loop()
            dataset[IMAGE_BYTES_COL_NAME] = loop.run_until_complete(
                load_image(dataset[CONSTANTS.IMAGE_PATH_COLUMN])
            )
            elapsed_time = time.process_time() - start_time
            logger.info("Downloaded assets in [{}] seconds".format(elapsed_time))
    else:
        logger.warning("Empty body passed, will return empty DataFrame")
    elapsed_time = time.process_time() - start_time
    logger.info("took [{}] seconds".format(elapsed_time))
    return dataset
def predict_fn(data, model_and_image_transformations):
    if data.empty == True:
        logger.warning("Empty data passed to predict_fn")
        return data

    data['prediction'] = data['_image_bytes_'].apply(lambda image_bytes: compute_embedding(model_and_image_transformations, image_bytes))
    return data

def compute_embedding(model_and_image_transformations, image_bytes):
    model, image_transformations = model_and_image_transformations
    try:
        if image_bytes.getbuffer().nbytes > 0:
            with Image.open(image_bytes) as image:
                batch = preprocess_data(image, image_transformations)
                batch = collate_fn(batch)
                with torch.inference_mode():  # don't calculate gradient
                    image_embeds = model(**batch).image_embeds
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # normalize before saving
                return image_embeds.cpu().detach().numpy().tolist()
    except Exception:
        logging.exception("Failed to open image.")
        pass

    return None

def output_fn(prediction, content_type) -> bytes:
    """
    The output_fn function is responsible for serialising the data that the predict_fn function returns as a prediction.
    This implementation serialises a batch of assets into a parquet file.
    """
    assert content_type in [
        "application/parquet"
    ], f"Requested unsupported ContentType in Accept: {content_type}"

    logger.debug("Generating parquet output")

    if prediction.empty is False:
        prediction.dropna(subset=[env.embedding_output_column_name], inplace=True)
        prediction.drop(IMAGE_BYTES_COL_NAME, axis=1, inplace=True)

    start_time = time.process_time()
    output_buffer = io.BytesIO()
    prediction.to_parquet(output_buffer, index=False, compression="snappy")
    elapsed_time = time.process_time() - start_time

    logger.info("Generated output in [{}] seconds".format(elapsed_time))
    return output_buffer.getvalue()

def preprocess_data(image_bytes,image_transformations: Transform):

    # use downloaded image
    images = [image_bytes]
    infer_dataset = []
    # transform image manually
    if all([image is None for image in images]):
        # flask.abort(400, 'Failed to download any of the images in the request.')
        logger.error('Failed to download any of the images in the request.')
    for image in images:
        skip = image is None
        if not skip:
            image = transforms.PILToTensor()(image)
            if image.shape[0] < 3:
                image = image.repeat(3, 1, 1)
            pix = image_transformations(image[:3, :, :])  # is a tensor
            infer_dataset.append(pix)
        else:
            null_image = torch.zeros(3, image_transformations.image_size, image_transformations.image_size)
            infer_dataset.append(null_image)
    return infer_dataset


def collate_fn(examples):
    pixel_values = torch.stack(examples)
    pixel_values = pixel_values.cuda() if torch.cuda.is_available() else pixel_values
    input_ids = torch.tensor([[1] for example in examples], dtype=torch.long)  # fake input_ids for model
    input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "return_loss": False
    }