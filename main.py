import argparse
import os
from typing import Tuple
import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
import sagemaker

from src.config.constants import CONSTANTS

DESTINATION_DATA_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "data")
DESTINATION_PROMPT_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "prompt")
DESTINATION_CODE_PATH = os.path.join(CONSTANTS.DEFAULT_DESTINATION_BASE_PATH, "code")
OWN_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(OWN_DIRECTORY_PATH, "src")
LLM_SOFT_LABELS_SCRIPT_PATH = os.path.join(SRC_PATH, "enrich_images", "enrich_images.py")
DEPLOY_MODEL_SCRIPT_PATH = os.path.join(SRC_PATH, "deployment", "deploy_model_job.py")


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
        # when data is contained in s3, we load it during job runtime instead of transfering
        # the files into the container. this is generally faster.
        inputs.append(
            ProcessingInput(
                source=args.data_path,
                destination=DESTINATION_DATA_PATH,
                input_name="data",
            ),
        )
    # Run the Processing job
    processor.run(
        code=LLM_SOFT_LABELS_SCRIPT_PATH,  # path to the .py file to run
        inputs=inputs,
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
    if args.command == "enrich":
        run_enrich_images_job(args)
    elif args.command == "deploy":
        run_deploy_model_job(args)
    else:
        print("Invalid command. Use 'enrich' or 'deploy'.")

if __name__ == "__main__":
    main()
