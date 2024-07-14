import os
from dataclasses import dataclass, field

import torch
from torch.utils.tensorboard import SummaryWriter
from config import constants
import logging
import os
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from torchvision import transforms
from datasets import Dataset, Image

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.utils import check_min_version, send_example_telemetry

from config.constants import CONSTANTS
from input_processing.process_text import tokenize_captions
from input_processing.collate import collate_fn
from finetune_clip.model_loader import ModelLoader
from utils.utils import load_image

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")


# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    text_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    vision_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    text_max_length: int = field(
        default=CONSTANTS.TEXT_MAX_LENGTH, metadata={
            "help": "max number of tokens for each captions. if caption holds less than this number, it will be padded."
                    "If the caption holds more tokens than this number, it will be truncated"}
    )
    data_parallel: bool = field(
        default=True,
        metadata={
            "help": "if to apply parallelization of the model on multiple GPUs."}
    )
    lit: bool = field(
        default=True,
        metadata={
            "help": "if to apply LiT while training, defaults to True. Freezes image encoder weights"}
    )


@dataclass
class FineTuneClIPJobConfig:
    dataset_path: str = field(
        default=None, metadata={
            "help": "Path to directory/S3 URI containing a DataFrame with image paths and captions"
        }
    )
    image_path_column: str = field(
        default=None, metadata={
            "help": "Name of the column containing image paths/URLs in the input dataframe"
        }
    )
    caption_column: str = field(
        default=CONSTANTS.CAPTION_COLUMN, metadata={
            "help": "Name of the column containing image URLs in the input dataframe, if a dataframe is provided instead of image files"
        }
    )
    random_seed: int = field(
        default=CONSTANTS.RANDOM_SEED, metadata={
            "help": "Random seed for reproducibility"}
    )


class FineTuneCLIPJob:
    def __init__(self, job_config: FineTuneClIPJobConfig, model_args: ModelArguments):
        self._job_config = job_config
        self._model_args = model_args
        with open(CONSTANTS.DEFAULT_TRAINING_ARGS_PATH, "r") as f:
            training_args_dict = json.load(f)
        self._training_args = TrainingArguments(
            **training_args_dict
        )

    def run(self):
        logging.info("Loading model...")
        model, tokenizer, image_processor, image_transformations = ModelLoader.load_model(
            vision_encoder_name_or_path=self._model_args.vision_model_name_or_path,
            text_encoder_name_or_path=self._model_args.text_model_name_or_path,
            data_parallel=self._model_args.data_parallel,
        )
        logging.info("Loading dataframe...")
        df = pd.read_parquet(self._job_config.dataset_path)
        logging.info(f"Dataframe loaded; number of training pairs: {df.shape[0]}")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=[self._job_config.image_path_column],
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": self._model_args.text_max_length
            },
            desc="Tokenizing captions"
        )

        def transform_images_batched(examples):
            images = [load_image(path) for path in examples[self._job_config.image_path_column]]
            images = [transforms.PILToTensor()(image) for image in images]
            examples[self._job_config.image_path_column] = [image_transformations(image) for image in images]
            return examples

        logging.info("Caption tokenization complete")
        dataset = dataset.cast_column(self._job_config.image_path_column, Image())
        dataset.set_transform(transform_images_batched, columns=[self._job_config.image_path_column],
                              output_all_columns=True)
        dataset = dataset.shuffle(seed=self._job_config.random_seed)
        splitted_dataset = dataset.train_test_split(train_size=0.8, seed=self._job_config.random_seed)
        output_dir = os.environ['SM_OUTPUT_DATA_DIR']
        log_dir = os.path.join(output_dir, "logs")
        writer = SummaryWriter(log_dir=log_dir)
        tensorboard_callback = transformers.integrations.TensorBoardCallback(writer)
        self._training_args.logging_dir = log_dir
        early_stop_callback = transformers.EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.0
        )
        trainer = Trainer(
            model=model,
            args=self._training_args,
            train_dataset=splitted_dataset["train"],
            eval_dataset=splitted_dataset["test"],
            data_collator=collate_fn,
            callbacks=[early_stop_callback, tensorboard_callback]
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        image_processor.save_pretrained(output_dir)
        writer.close()
        logging.info(f"Training complete; Model, tokenizer and image processor saved to {output_dir}")


if __name__ == '__main__':
    hyperparameters = json.loads(os.environ['SM_HPS'])
    model_args = ModelArguments(
        text_model_name_or_path=hyperparameters["text_model_name_or_path"],
        vision_model_name_or_path=hyperparameters["vision_model_name_or_path"],
        text_max_length=hyperparameters["text_max_length"],
        data_parallel=hyperparameters["data_parallel"],
        lit=hyperparameters["lit"]
    )
    job_config = FineTuneClIPJobConfig(
        dataset_path=hyperparameters["dataset_path"],
        image_path_column=hyperparameters["image_path_column"],
        caption_column=hyperparameters["caption_column"],
        random_seed=hyperparameters["random_seed"]
    )
    logger.info(f"starting training with model args: {model_args}; and with job config: {job_config}")
    FineTuneCLIPJob(job_config, model_args).run()
