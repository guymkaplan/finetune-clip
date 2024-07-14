# Image Enrichment Component

This folder contains the image enrichment component of the Fine-tune CLIP project. It's responsible for generating captions for images using various language models.

## Table of Contents

1. [Overview](#overview)
2. [Key Files](#key-files)
3. [Registering a New Model](#registering-a-new-model)
4. [Configuration](#configuration)
5. [Prompt Template](#prompt-template)
6. [Chain-of-Thought (CoT) Prompting](#chain-of-thought-cot-prompting)
7. [How It Works](#how-it-works)

## Overview

The image enrichment component is designed to process a large number of images concurrently, generating captions using a specified language model. It leverages multiprocessing for efficient execution and is intended to run as a SageMaker job.

## Key Files

- `enrich_images_job.py`: Main script for the image enrichment job.
- `models.py`: Contains the model registration system and factory method.
- `enrich_image_model.py`: Base class for image enrichment models.
- `prompt_template.py`: Handles the prompt templates for caption generation.

## Registering a New Model

To add support for a new image enrichment model:

1. Create a new class that inherits from `EnrichImageModel` in `enrich_image_model.py`.
2. Implement the required methods, especially `infer()`.
3. Use the `@register_model` decorator to register your new model.

Example:

```python
from models import register_model
from enrich_image_model import EnrichImageModel
from prompt_template import PromptTemplate
from PIL import Image
from typing import EnrichImageResult

@register_model("my_new_model")
class MyNewModel(EnrichImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here

    def infer(self, prompt: PromptTemplate, image: Image.Image, image_name: str) -> EnrichImageResult:
        # Implement the inference logic
        pass
```

After registration, you can use your new model by setting the `MODEL_NAME` environment variable to "my_new_model" when running the SageMaker job.

## Configuration

The `EnrichImagesJobConfig` dataclass in `enrich_images_job.py` defines the configuration options for the enrichment job. Key options include:

- `data_path`: Path to image files or a dataframe with image URLs.
- `prompt_path`: Path to the JSON file containing the prompt template.
- `output_path`: S3 path to save the generated captions.
- `num_of_threads`: Number of threads for parallel processing.
- `image_target_size`: Target size for image processing.

## Prompt Template

The `PromptTemplate` class is used to structure the prompts for the language model. It includes the following components:

- `system_prompt`: Sets the context for the AI assistant.
- `prompt_prefix`: Text that comes before the image description.
- `prompt_suffix`: Text that comes after the image description.
- `regex`: A regular expression pattern to extract the desired output.
- `cot_prompt`: An optional Chain-of-Thought prompt (see below).

The `PromptTemplate` can be loaded from a JSON file using the `from_json` class method. Here's an example of a prompt template JSON file:

```json
{
  "system_prompt": "You are a helpful AI assistant. Wrap your final answer around the  tag.",
  "prompt_prefix": "Describe the following image in detail:",
  "prompt_suffix": "Please provide a detailed caption for the image. \nQ:",
  "cot_prompt": {
    "image_path": "example_data/cot_image_example.jpg",
    "explanation": "A: \"Park moments: Dad, baby, and their furry friend\":\n\n    Conciseness: The user requested a shorter caption, so I aimed to keep it brief while still capturing the essence of the image.\n    Key Elements: The image features a man, a baby, and a dog in a park setting. It's important to mention all three main subjects to give a full picture.\n    Warmth and Connection: \"Park moments\" conveys a sense of shared experience and leisure, highlighting the bond between the individuals and their pet.\n    Simplicity and Clarity: The caption is straightforward and easy to understand, which makes it effective and appealing for a wide audience.\n    Smartness: The caption subtly reflects the scene's emotions and setting without being overly descriptive, maintaining a balance between being informative and evocative."
  },
  "regex": ".*"
}
```

## Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting is a technique used to improve the performance of language models by encouraging them to break down complex tasks into steps. In the context of image captioning, CoT prompting can help the model generate more accurate and relevant captions.

The `CoTPrompt` class in the `PromptTemplate` includes:

- `image`: An optional example image.
- `explanation`: A detailed explanation of the thought process for generating a caption.

Users are encouraged to customize the CoT prompt based on their specific use case. For example, if the goal is to classify car types, the prompt should guide the model to focus on identifying and describing the car in the image.

As researched in the paper "State of What Art? A Call for Multi-prompt LLM Evaluation", users are encouraged to experiment with different prompts to find the most effective one for their specific task and dataset.

## How It Works

1. The job loads image paths from the specified `data_path`.
2. It creates a multiprocessing pool to process images in parallel.
3. For each image, it:
   - Loads the image
   - Queries the specified model using the prompt template
   - Stores the result (image path, caption, and full LLM response)
4. Results are combined into a single DataFrame and saved as a Parquet file in the specified `output_path`.

Note: This job is designed to run on SageMaker and should not be executed locally. The environment variables required for execution are set up automatically in the SageMaker environment.