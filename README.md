# Fine-tune CLIP Project

This project demonstrates an advanced pipeline for enriching images with captions, fine-tuning a CLIP (Contrastive Language-Image Pre-training) model, and deploying the model using Amazon SageMaker. It showcases various high-performance computing techniques and offers flexibility in model selection and deployment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
   - [Enriching Images](#enriching-images)
   - [Fine-tuning CLIP](#fine-tuning-clip)
   - [Deploying the Model](#deploying-the-model)
7. [Advanced Features](#advanced-features)
8. [Contributing](#contributing)
9. [License](#license)
10. [TODO](#todo)

## Project Overview

This project consists of three main components:

1. **Image Enrichment**: A process to generate captions for images using a language model.
2. **CLIP Fine-tuning**: Fine-tuning a CLIP model on a custom dataset of images and captions.
3. **Model Deployment**: Deploying the fine-tuned model to Amazon SageMaker for inference.

## Key Features

- Multithreading for efficient image processing and model inference
- Data parallelization across multiple GPUs within a single instance
- Flexible LLM integration for image enrichment tasks
- Customizable endpoint deployment with user-defined code
- Scalable architecture leveraging Amazon SageMaker

## Project Structure

```
finetune-clip
├── README.md
├── __init__.py
├── main.py
└── src
    ├── config
    ├── deployment
    ├── enrich_images
    ├── finetune_clip
    ├── input_processing
    └── utils
```

## Requirements

- Python 3.7+
- AWS account with SageMaker access
- Boto3
- SageMaker Python SDK
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/guymkaplan/finetune-clip.git
   cd finetune-clip
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your AWS credentials and configure your AWS CLI.

## Usage

The project can be run using the `main.py` script, which provides three main commands: `enrich`, `fine-tune`, and `deploy`.

### Enriching Images

To enrich images with captions:

```
python main.py enrich --data_path <path_to_images> --prompt_path <path_to_prompt_json> --output_path <s3_output_path>
```

Optional arguments:
- `--num_of_threads`: Number of threads for model querying
- `--image_target_size`: Target size for image processing (default: 224x224)
- `--instance_type`: SageMaker instance type for processing (default: ml.r5.16xlarge)
- `--docker_image`: Docker image to use for processing

### Fine-tuning CLIP

To fine-tune the CLIP model, provide:
- text model stored in S3 or model identifier from [huggingface](huggingface.co/models)
- vision model stored in S3 or model identifier from [huggingface](huggingface.co/models)
- dataset in S3 as parquet where each sample has a "caption" and "image_path" column:
  - "caption": text that describes the image
  - "image_path": url to image or "opt/ml/input/data/images/{image_name}.{image_extension}" in case where `images_path` is provided. 

```
python main.py fine-tune --text_model_name_or_path <path_or_name> --vision_model_name_or_path <path_or_name> --dataset_path <s3_path_to_dataset_parquet> --output_path <s3_output_model_path>
```

Optional arguments:
- `--text_max_length`: Max token length for captions
- `--data_parallel`: Enable data parallelism
- `--lit`: Apply [LiT technique](https://arxiv.org/pdf/2111.07991) by freezing the vision model while training 
- `--images_path`: S3 path to images. If using this input, \"image_path\" column in dataset should be "opt/ml/input/data/images/{image_name}.{image_extension}"
- `--image_path_column`: Column name for image paths in dataset (default: "image_path").
- `--caption_column`: Column name for captions in dataset
- `--random_seed`: Random seed for reproducibility

### Deploying the Model

To deploy the fine-tuned model:

```
python main.py deploy --model_type <model_type: vision | text> --model_name <model_name> --model_uri <s3_model_uri> --endpoint_name <endpoint_name>
```

Optional arguments:
- `--main_module_name`: Name of the main module for the model
- `--instance_type`: Instance type for deployment job
- `--instance_type_endpoint`: Instance type for SageMaker endpoint
- `--initial_instance_count_endpoint`: Initial instance count for endpoint
- `--deploy_model_as_endpoint`: Flag to deploy as endpoint
- `--usr_code_dir`: Directory with user code for the model
- `--docker_image`: Docker image for processing

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.


## TODO
- Optimize hyperparameters for CLIP fine-tuning
- Add support for multiple image enrichment models
- Add support for distributed training across multiple instances
- Support for batch transform jobs on an existing model