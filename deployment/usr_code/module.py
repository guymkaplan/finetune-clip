import os
import json
import torch
from transformers import VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor
import logging

from input_processing.process_images import ImageTransformation
from utils.utils import load_image

INPUT_TEXT_FIELD_NAME = os.environ.get(
    "INPUT_TEXT_FIELD_NAME", "caption"
)
INPUT_IMAGE_URL_FIELD_NAME = os.environ.get(
    "INPUT_IMAGE_URL_FIELD_NAME", "image_url"
)

logger = logging.getLogger(__name__)

def model_fn(model_dir,device):
    """
    Load the model from the directory where SageMaker has downloaded the model artifacts.
    """
    logging.info("loading model....")
    state = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
    model = VisionTextDualEncoderModel.from_pretrained(model_dir)
    model.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info(f"Loaded model to gpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    image_transformations = ImageTransformation(
        model.config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std
    )
    return model, tokenizer, image_transformations

def input_fn(request_body, request_content_type):
    """
    Process the input data.
    """
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        if INPUT_TEXT_FIELD_NAME not in request and INPUT_IMAGE_URL_FIELD_NAME not in request:
            raise ValueError(
                f"Missing required field: either {INPUT_TEXT_FIELD_NAME} or {INPUT_IMAGE_URL_FIELD_NAME} must be provided")
        if INPUT_TEXT_FIELD_NAME in request and INPUT_IMAGE_URL_FIELD_NAME in request:
            raise ValueError(
                f"Only one of {INPUT_TEXT_FIELD_NAME} or {INPUT_IMAGE_URL_FIELD_NAME} should be provided, not both")

        if INPUT_TEXT_FIELD_NAME in request:
            return {"type": "text", "data": request[INPUT_TEXT_FIELD_NAME]}
        else:
            return {"type": "image", "data": request[INPUT_IMAGE_URL_FIELD_NAME]}
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_tokenizer_transformation):
    """
    Perform inference on the input data using the model and tokenizer.
    """
    model, tokenizer, image_transformations = model_tokenizer_transformation

    if input_data["type"] == "text":
        inputs = tokenizer(input_data["data"], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
    elif input_data["type"] == "image":
        image_url = input_data["data"]
        image = load_image(image_url)
        inputs = image_transformations(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
    else:
        raise ValueError(f"Unsupported input type: {input_data['type']}")
    outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # normalize
    return outputs

def output_fn(prediction, response_content_type):
    """
    Format the prediction output.
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction.cpu().numpy().tolist())
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")