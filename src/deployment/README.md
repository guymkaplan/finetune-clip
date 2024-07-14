### Deploying the Model

The deployment process is handled by the `deploy_model_job.py` script, which prepares the model for deployment and creates a SageMaker endpoint or model.

To deploy the fine-tuned model:

```
python main.py deploy --model_type <model_type: vision | text> --model_name <model_name> --model_uri <s3_model_uri> --endpoint_name <endpoint_name>
```

Depending on the arguments, theis job will deploy either a text tower or a vision tower, which can either be deployed as endpoints or only contain the model.

Optional arguments:
- `--main_module_name`: Name of the main module for the model
- `--instance_type`: Instance type for the SageMaker endpoint (default: ml.g5.8xlarge)
- `--initial_instance_count_endpoint`: Initial instance count for endpoint (default: 1)
- `--deploy_model_as_endpoint`: Flag to deploy as endpoint (default: False)
- `--usr_code_dir`: Directory with user code for the model
- `--docker_image`: Docker image for processing

#### Deployment Process

1. The script prepares the model for deployment by:
   - Downloading the model artifact from S3
   - Extracting the model files
   - Adding user code to the model package
   - Uploading the packaged model back to S3

2. A PyTorch model is created using the SageMaker PyTorchModel class, specifying:
   - Model name
   - IAM role for execution
   - Entry point script (`module.py`)
   - Source directory (user code directory)
   - Model data location (S3 URI of the packaged model)
   - Docker image for inference

3. Depending on the `deploy_model_as_endpoint` flag:
   - If True: The model is deployed as a SageMaker endpoint
   - If False: The model is created in SageMaker but not deployed as an endpoint

The deployment process uses the following Docker image for inference:
`763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04`

#### Environment Variables

The deployment script uses the following environment variables:

- `MODEL_ENTRY_POINT`: Location of endpoint script. Derived from `<model_type>` argument 
- `MODEL_NAME`: Name of the model to be deployed
- `MODEL_URI`: S3 URI of the trained model artifact
- `MAIN_MODULE_NAME`: Name of the main module for the model
- `ENDPOINT_NAME`: Name of the SageMaker endpoint
- `INSTANCE_TYPE`: Instance type for the SageMaker endpoint (default: ml.g5.8xlarge)
- `INITIAL_INSTANCE_COUNT`: Initial number of instances for the SageMaker endpoint (default: 1)
- `DEPLOY_MODEL_AS_ENDPOINT`: Flag to indicate whether to deploy the model as an endpoint (default: False)
- `USR_CODE_DIR`: Directory containing the user code for the model

These environment variables are set by the `main.py` script based on the command-line arguments provided.