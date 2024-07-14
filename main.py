import argparse
import sys
import os
from typing import Tuple
import boto3
from sagemaker.processing import ProcessingInput, ScriptProcessor
from sagemaker.inputs import TrainingInput
import sagemaker
from sagemaker.estimator import Estimator
from src.config.constants import CONSTANTS
import logging

DESTINATION_DATA_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "data")
DESTINATION_PROMPT_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "prompt")
DESTINATION_CODE_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "code")
OWN_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(OWN_DIRECTORY_PATH, "src")
LLM_SOFT_LABELS_SCRIPT_PATH = os.path.join(SRC_PATH, "enrich_images", "enrich_images.py")
DEPLOY_MODEL_SCRIPT_PATH = os.path.join(SRC_PATH, "deployment", "deploy_model_job.py")
FINETUNE_CLIP_SCRIPT_PATH = os.path.join(SRC_PATH, "finetune_clip", "finetune_clip.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich images with captions or deploy model")
    subparsers = parser.add_subparsers(dest="command", help="Available commands: enrich | deploy | fine-tune")
    # Enrich images command
    enrich_parser = subparsers.add_parser("enrich", help="Enrich images with captions")
    enrich_parser.add_argument("--data_path", type=str, required=True,
                               help="Path to directory containing image files, or path to directory holding a dataframe with image urls, or S3 URI to a dataframe with image urls.")
    enrich_parser.add_argument("--prompt_path", type=str, required=True,
                               help="Path to the prompt in .json format. The json must hold the following keys: `system_prompt`, `prompt_prefix`, `prompt_suffix`, `regex`. Optional key: `cot_prompt` holding `image_path` and `explanation` keys.")
    enrich_parser.add_argument("--output_path", type=str, required=True,
                               help="Path to save captions in S3, like so: s3://my-heartwarming-bucket/training_data/")
    enrich_parser.add_argument("--num_of_threads", type=int, default=None,
                               help="Number of threads that query the model")
    enrich_parser.add_argument("--image_target_size", type=Tuple[int, int], default=(224, 224),
                               help="The pixel size that will be used when querying the model. Note that larger values require more compute, potentially increasing costs. For large images with fine details, it is recommended to use maximum image size.")
    enrich_parser.add_argument("--instance_type", type=str, default=CONSTANTS.ENRICH_IMAGES_DEFAULT_INSTANCE_TYPE,
                               help="Type of instance to use for processing")
    enrich_parser.add_argument("--docker_image", type=str, default=CONSTANTS.DEFAULT_CONTAINER_IMAGE,
                               help="Docker image to use for processing")
    # Deploy model command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model")
    deploy_parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be deployed")
    deploy_parser.add_argument("--model_uri", type=str, required=True, help="S3 URI of the trained model artifact")
    deploy_parser.add_argument("--main_module_name", type=str, help="Name of the main module for the model")
    deploy_parser.add_argument("--endpoint_name", type=str, help="Name of the SageMaker endpoint")
    deploy_parser.add_argument("--instance_type", type=str, default="ml.m5.large",
                               help="Instance type for the deployment job")
    deploy_parser.add_argument("--instance_type_endpoint", type=str, default="ml.g5.8xlarge",
                               help="Instance type for the SageMaker endpoint. NOTE: this is not for the deploy model job, but for the endpoint the model will be in")
    deploy_parser.add_argument("--initial_instance_count_endpoint", type=int, default=1,
                               help="Initial number of instances for the SageMaker endpoint")
    deploy_parser.add_argument("--deploy_model_as_endpoint", action="store_true",
                               help="Flag to indicate whether to deploy the module as an endpoint")
    deploy_parser.add_argument("--usr_code_dir", type=str,
                               default=os.path.join(SRC_PATH, CONSTANTS.DEFAULT_USER_DIR_PATH),
                               help="Directory containing the user code for the model")

    # Fine-tune CLIP command
    # Model arguments:
    finetune_parser = subparsers.add_parser("fine-tune", help="Fine-tune CLIP model")

    finetune_parser.add_argument("--text_model_name_or_path", type=str, required=True,
                                 help="Path to pretrained text model in s3 or model identifier from huggingface.co/models")
    finetune_parser.add_argument("--vision_model_name_or_path", type=str, required=True,
                                 help="Path to pretrained vision model in s3 or model identifier from huggingface.co/models")
    finetune_parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                                 help="Pretrained tokenizer name or path if not the same as model_name")
    finetune_parser.add_argument("--image_processor_name", type=str, default=None,
                                 help="Name or path of preprocessor config.")
    finetune_parser.add_argument("--text_max_length", type=int, default=CONSTANTS.TEXT_MAX_LENGTH,
                                 help="Max number of tokens for each caption. If caption holds less than this number, it will be padded. If the caption holds more tokens than this number, it will be truncated")
    finetune_parser.add_argument("--data_parallel", type=bool, default=True,
                                 help="If to apply parallelization of the model on multiple GPUs")
    finetune_parser.add_argument("--lit", type=bool, default=True,
                                 help="If to apply LiT while training. Defaults to True. Freezes image encoder weights")

    # Fine-tune job config:
    finetune_parser.add_argument("--dataset_path", type=str, required=True,
                                 help="Path to directory/S3 URI containing a DataFrame with image paths and captions")
    finetune_parser.add_argument("--images_path", type=str,
                                 help="Path to S3 where images are stored. If using this input, \"image_path\" column in dataset should be f\"opt/ml/input/data/images/{image_name}.{image_extension}\"")
    finetune_parser.add_argument("--image_path_column", type=str, default=CONSTANTS.IMAGE_PATH_COLUMN,
                                 help="Name of the column containing image paths/URLs in the input dataframe")
    finetune_parser.add_argument("--caption_column", type=str, default=CONSTANTS.CAPTION_COLUMN,
                                 help="Name of the column containing captions in the input dataframe")
    finetune_parser.add_argument("--output_path", type=str, required=True,
                                 help="S3 path to save the fine-tuned model")
    finetune_parser.add_argument("--random_seed", type=int, default=CONSTANTS.RANDOM_SEED,
                                 help="Random seed for reproducibility")
    return parser.parse_args()

def create_enrich_images_env(args):
    env = {
        "DATA_PATH": args.data_path,
        "IMAGE_URL_COLUMN": args.image_url_column,
        "PROMPT_PATH": DESTINATION_PROMPT_PATH,
        "OUTPUT_PATH": DESTINATION_DATA_PATH,
        "IMAGE_TARGET_SIZE": str(args.image_target_size),
    }
    if args.num_of_threads is not None:
        env["NUM_OF_THREADS"] = str(args.num_of_threads)
    return env


def run_enrich_images_job(args):
    processor = ScriptProcessor(
        max_runtime_in_seconds=3600,
        image_uri=args.docker_image,
        instance_type=args.instance_type,
        sagemaker_session=sagemaker.Session(boto3.Session()),
        instance_count=1,
        base_job_name="enrich-images",
        role=sagemaker.get_execution_role(),
        command=["python3"],
        env=create_enrich_images_env(args)
    )
    inputs = [
        ProcessingInput(
            input_name="code", source=SRC_PATH, destination=DESTINATION_CODE_PATH
        ),
        ProcessingInput(
            source=args.prompt_path,
            destination=DESTINATION_PROMPT_PATH,
            input_name="prompt",
        )
    ]
    if not args.data_path.startswith("s3://"):
        # when data is contained in s3, we load it during job runtime instead of transferring
        # the files into the container. this is generally faster.
        inputs.append(
            ProcessingInput(
                source=args.data_path,
                destination=DESTINATION_DATA_PATH,
                input_name="data",
            ),
        )
    processor.run(
        code=LLM_SOFT_LABELS_SCRIPT_PATH,
        inputs=inputs,
    )

def run_finetune_clip_job(args):
    sagemaker_session = sagemaker.Session(boto3.Session())
    role = sagemaker.get_execution_role()

    estimator = Estimator(
        image_uri=CONSTANTS.DEFAULT_CONTAINER_IMAGE,
        dependencies=[SRC_PATH],
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        entry_point=FINETUNE_CLIP_SCRIPT_PATH,
        output_path=args.output_path,
    )
    inputs = dict()
    hyper_params = dict(
        text_model_name_or_path=args.text_model_name_or_path,
        vision_model_name_or_path=args.vision_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        image_processor_name=args.image_processor_name,
        text_max_length=args.text_max_length,
        data_parallel=args.data_parallel,
        lit=args.lit,
        # FineTuneClIPJobConfig
        dataset_path=args.dataset_path,
        images_path=args.images_path,
        image_path_column=args.image_path_column,
        caption_column=args.caption_column,
        random_seed=args.random_seed,
    )
    if args.images_path:  # case where the images are stored in s3
        inputs["images"] = TrainingInput(args.images_path, distribution="FullyReplicated")
    if args.vision_model_name_or_path.startswith("s3://"):
        # insert the pretrained model as training input:
        inputs["vision_model"] = TrainingInput(
            args.vision_model_name_or_path, distribution="FullyReplicated"
        )
        # point to where the pretrained model will be stored in the training container
        hyper_params["vision_model_name_or_path"] = CONSTANTS.DEFAULT_PRETRAINED_VISION_MODEL_PATH
    if args.text_model_name_or_path.startswith("s3://"):
        # insert the pretrained model as training input:
        inputs["text_model"] = TrainingInput(
            args.text_model_name_or_path, distribution="FullyReplicated"
        )
        # point to where the pretrained model will be stored in the training container
        hyper_params["text_model_name_or_path"] = CONSTANTS.DEFAULT_PRETRAINED_TEXT_MODEL_PATH
    estimator.set_hyperparameters(**hyper_params)
    estimator.fit(
        inputs=inputs,
        job_name=f"finetune-clip-{sagemaker_session.default_bucket()}",
    )
def create_deploy_model_env(args):
    return {
        "MODEL_NAME": args.model_name,
        "MODEL_URI": args.model_uri,
        "MAIN_MODULE_NAME": args.main_module_name,
        "ENDPOINT_NAME": args.endpoint_name,
        "INSTANCE_TYPE": args.instance_type_endpoint,
        "INITIAL_INSTANCE_COUNT": str(args.initial_instance_count),
        "DEPLOY_MODEL_AS_ENDPOINT": str(args.deploy_model_as_endpoint),
        "USR_CODE_DIR": args.usr_code_dir,
    }


def run_deploy_model_job(args):
    processor = ScriptProcessor(
        max_runtime_in_seconds=3600,
        image_uri=args.docker_image,
        instance_type=args.instance_type,
        sagemaker_session=sagemaker.Session(boto3.Session()),
        instance_count=1,
        base_job_name="deploy-model",
        role=sagemaker.get_execution_role(),
        command=["python3"],
        env=create_deploy_model_env(args)
    )

    inputs = [
        ProcessingInput(
            input_name="code",
            source=SRC_PATH,
            destination=DESTINATION_CODE_PATH
        ),
        ProcessingInput(
            input_name="usr_code",
            source=args.usr_code_dir,
            destination=os.path.join(DESTINATION_CODE_PATH, "usr_code")
        )
    ]

    processor.run(
        code=DEPLOY_MODEL_SCRIPT_PATH,
        inputs=inputs,
    )


def main():
    args = parse_args()
    logging.info(f"Received arguments: {args}")
    if args.command == "enrich":
        run_enrich_images_job(args)
    elif args.command == "deploy":
        run_deploy_model_job(args)
    elif args.command == "fine-tune":
        run_finetune_clip_job(args)
    else:
        logging.info("Invalid command. Use 'enrich', 'deploy' or 'fine-tune'.")

if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Create a StreamHandler that writes to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    root_logger.addHandler(stdout_handler)
    main()
