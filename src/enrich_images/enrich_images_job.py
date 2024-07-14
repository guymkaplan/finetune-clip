import logging
import multiprocessing
import os
import pandas as pd
from dataclasses import dataclass, field

from config.constants import CONSTANTS

from prompt_template import PromptTemplate
from models import build_model
from ..utils.utils import load_image
logger = logging.getLogger(__name__)


@dataclass
class EnrichImagesJobConfig:
    data_path: str = field(
        default=None, metadata={
            "help": "Path to directory containing image files, or path to directory holding a dataframe with image urls, or S3 URI to a dataframe with image urls."
        }
    )
    image_url_column: str = field(
        default="url", metadata={
            "help": "Name of the column containing image URLs in the input dataframe, if a dataframe is provided instead of image files"
        }
    )
    prompt_path: str = field(
        default=None, metadata={"help": "Path to the prompt in .json format. The json must hold the following keys:"
                                        "`system_prompt`, `prompt_prefix`, `prompt_suffix`, `regex`. Optional key: `cot_prompt` holding `image_path` and `explanation` keys."}
    )
    output_path: str = field(
        default=None,
        metadata={
            "help": "Path to save captions in S3, like so: s3://my-heartwarming-bucket/training_data/"}
    )
    num_of_threads: int = field(
        default=os.cpu_count(), metadata={
            "help": "number of threads that query the model"
        }
    )
    image_target_size: tuple[int, int] = field(
        default=CONSTANTS.DEFAULT_IMAGE_SIZE,
        metadata={
            "help": "The pixel size that will be used when querying the model."
                    "Note that larger values require more compute, potentially increasing costs."
                    "For large images with fine details, it is recommended to use maximum image size."
                    "e.g. for Claude3, see https://docs.anthropic.com/en/docs/vision"
        }
    )


class EnrichImagesJob:
    def __init__(self, config: EnrichImagesJobConfig):
        self._prompt_template = PromptTemplate.from_json(config.prompt_path)
        self._config = config
        self._model = build_model(model_name=os.environ['MODEL_NAME'])

    def run(self):
        image_paths = self._load_image_paths()
        with multiprocessing.Pool(processes=self._config.num_of_threads) as pool:
            results = pool.map(self._query_model, image_paths)
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_parquet(f"{self._config.output_path}output.parquet")

    def _load_image_paths(self):
        """Load image paths from a directory or a dataframe."""
        if os.path.isdir(self._config.data_path) and not self._config.data_path.startswith("s3://"):
            images = set()
            for filename in os.listdir(self._config.data_path):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    images.add(os.path.join(self._config.data_path, filename))
        else:  # is local path or s3 uri to dataframe
            df = pd.read_parquet(self._config.data_path)
            images = set(df[self._config.image_url_column].unique())
        return images

    def _query_model(self, image_path: str):
        image = load_image(image_path)
        try:
            result = self._model.infer(prompt=self._prompt_template, image=image, image_name=image_path)
            return pd.DataFrame({
                CONSTANTS.IMAGE_PATH_COLUMN: [result.image_name],
                CONSTANTS.CAPTION_COLUMN: [result.caption],
                CONSTANTS.LLM_RESPONSE_COLUMN: [result.llm_response]
            })
        except:
            logger.error(f"failed creating a caption for the image in {image_path}")
            return


if __name__ == '__main__':
    data_path = os.environ.get("DATA_PATH")
    image_url_column = os.environ.get("IMAGE_URL_COLUMN", "url")  # Default to "url" if not set
    prompt_path = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "prompt")
    output_path = os.environ.get("OUTPUT_PATH")
    image_target_size = tuple(int(x) for x in os.environ.get("IMAGE_TARGET_SIZE").strip("()").split(","))
    config = EnrichImagesJobConfig(
        data_path=data_path,
        image_url_column=image_url_column,
        prompt_path=prompt_path,
        output_path=output_path,
        image_target_size=image_target_size,
    )
    logger.info(f"Starting job with config: {config}")
    EnrichImagesJob(config=config).run()
    logger.info("Job completed successfully!")