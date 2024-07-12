import logging
import sys
from dataclasses import field, dataclass
from urllib.parse import urlparse

import sagemaker
import os
import tempfile
import boto3
import tarfile
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
sys.path.append("usr_code")


@dataclass
class DeployModelJobConfig:
    model_name: str = field(
        default=None, metadata={
            "help": "name of the model to be deployed"}
    )
    model_uri: str = field(
        default=None, metadata={
            "help": "S3 URI of the trained model artifact"}
    )
    main_module_name: str = field(
        default=None, metadata={
            "help": "name of the main module for the model"}
    )
    endpoint_name: str = field(
        default=None, metadata={
            "help": "name of the SageMaker endpoint"}
    )
    instance_type: str = field(
        default="ml.g5.8xlarge", metadata={
            "help": "instance type for the SageMaker endpoint"}
    )
    initial_instance_count: int = field(
        default=1, metadata={
            "help": "initial number of instances for the SageMaker endpoint"}
    )
    deploy_model_as_endpoint: bool = field(
        default=False, metadata={
            "help": "flag to indicate whether to deploy the module as an endpoint. if False, will only craete the sagemaker model and will not be deployed as an endpoint"}
    )
    usr_code_dir: str = field(
        default="", metadata={
            "help": "directory containing the user code for the model"}
    )


class DeployModelJob():
    def __init__(
            self,
            config: DeployModelJobConfig,
    ) -> None:
        super(DeployModelJob, self).__init__()
        self._config = config

    def _prepare_model_for_deployment(self, model_name, model_uri):
        s3 = boto3.client('s3')
        output_artifact_bucket = sagemaker.session.Session().default_bucket()
        parsed_uri = urlparse(model_uri)
        if parsed_uri.scheme != 's3':
            raise ValueError("Not an S3 URI")
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip('/')
        with tempfile.TemporaryDirectory(prefix='image-text-embedding-model-') as tmp_model_data_extract_dir:
            logging.info(f"Using temporary directory: {tmp_model_data_extract_dir}")
            model_version = key.split("/")[0]
            temp_path = f"{tmp_model_data_extract_dir}/{model_version}"
            os.mkdir(temp_path)
            model_path = f"{temp_path}/model.tar.gz"
            with open(model_path, 'wb') as f:
                s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)
                logging.info("Saved model artifact into: {}".format(model_path))
            with tarfile.open(model_path, "r:gz") as f:
                logging.info("Extracting model...")
                f.extractall(tmp_model_data_extract_dir)
            with tarfile.open(f"model.tar.gz", "w:gz") as output_model_data:
                for filename in os.listdir(tmp_model_data_extract_dir):
                    output_model_data.add(os.path.join(tmp_model_data_extract_dir, filename), arcname=filename)
                for filename in os.listdir(self._config.usr_code_dir):
                    output_model_data.add(os.path.join(self._config.usr_code_dir, filename), arcname='usr/' + filename)

            s3.upload_file(f"model.tar.gz", output_artifact_bucket, f"models/{model_name}.tar.gz")
            logging.info("Uploaded artifact into: s3://{}/{}".format(output_artifact_bucket, f"models/{model_name}.tar.gz"))
            return "s3://{}/{}".format(output_artifact_bucket, f"models/{model_name}.tar.gz")

    def _create_pytorch_model(self, model_name, entry_point, source_dir, model_uri):
        return PyTorchModel(
            name=model_name,
            role=get_execution_role(),
            entry_point=entry_point,
            source_dir=source_dir,
            model_data=model_uri,
            image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
        )

    def run(self):
        endpoint_name = f'{self._config.model_name}-endpoint'
        packaged_model_uri = self._prepare_model_for_deployment(model_uri=self._config.model_uri,
                                                                model_name=self._config.model_name)
        model = self._create_pytorch_model(model_name=self._config.model_name,
                                           entry_point="module.py",
                                           source_dir=self._config.usr_code_dir,
                                           model_uri=packaged_model_uri
                                           )
        if self._config.deploy_model_as_endpoint:
            model.deploy(
                instance_type=self._config.instance_type,
                initial_instance_count=self._config.initial_instance_count,
                endpoint_name=endpoint_name)
            logging.info(f"Deployed model as endpoint: {endpoint_name}")
        else:
            model.create(
                instance_type=self._config.instance_type,
                initial_instance_count=self._config.initial_instance_count
            )
            logging.info(f"Deployed model: {self._config.model_name}")

if __name__ == '__main__':
    model_name = os.environ.get("MODEL_NAME")
    model_uri = os.environ.get("MODEL_URI")
    main_module_name = os.environ.get("MAIN_MODULE_NAME")
    endpoint_name = os.environ.get("ENDPOINT_NAME")
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.g5.8xlarge")
    initial_instance_count = int(os.environ.get("INITIAL_INSTANCE_COUNT", 1))
    deploy_model_as_endpoint = bool(os.environ.get("DEPLOY_MODEL_AS_ENDPOINT", False))
    usr_code_dir = os.environ.get("USR_CODE_DIR")
    config = DeployModelJobConfig(
        model_name=model_name,
        model_uri=model_uri,
        main_module_name=main_module_name,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        deploy_model_as_endpoint=deploy_model_as_endpoint,
        usr_code_dir=usr_code_dir
    )
    logging.info(f"Starting job with config: {config}")
    DeployModelJob(config=config).run()
    logging.info("Job completed successfully!")