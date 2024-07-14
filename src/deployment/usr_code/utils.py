import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import validators
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_fixed

import torch
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, ConvertImageDtype, Normalize

from config.constants import CONSTANTS

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
async def download_image(session, url, rate_limiter):
    if validators.url(url) is True:
        logger.info(f"Downloading image from URL: {url}...")
        start_time = time.process_time()

        async with rate_limiter, session.get(url) as response:
            elapsed_time = time.process_time() - start_time
            logger.debug("... image from URL in [{}] seconds".format(elapsed_time))

            response_status_code = response.status
            response_content_type = response.headers["Content-Type"]

            if response_status_code == 200 and response_content_type.startswith("image"):
                image_bytes = io.BytesIO(await response.read())
                image_bytes_nbytes = image_bytes.getbuffer().nbytes
                if image_bytes_nbytes <= CONSTANTS.DEFAULT_IMAGE_MAX_BYTES:
                    try:
                        with Image.open(image_bytes) as image:
                            return image_bytes
                    except Exception as e:
                        logger.error(f"Failed to open image: {e}")
            else:
                logger.warning(
                    "URL=[{}] returned status code=[{}] and Content-Type=[{}]".format(
                        url, response_status_code, response_content_type
                    )
                )

    logger.warning("Invalid URL or image: [{}]".format(url))
    return io.BytesIO()


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=77,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class InferenceArguments:
    fp16: bool = field(
        default=False, metadata={'help': 'cast to fp16 for inference'}
    )
    batch_size: Optional[int] = field(
        default=64, metadata={"help": "Inference time batch size."}
    )
    max_seq_length: Optional[int] = field(
        default=77, metadata={"help": "Maximum input sequence length for text encoder."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
