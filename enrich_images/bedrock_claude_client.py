import base64
import json
import logging
import re
from io import BytesIO

from PIL import Image
from prompt_template import PromptTemplate
from enrich_image_model import EnrichImageModel, EnrichImageResult
import boto3
from utils import retry_exp_backoff

logger = logging.getLogger(__name__)
CONFIG_PATH = 'enrich_images/bedrock_claude_config.json'

class BedrockClaudeSonnet(EnrichImageModel):
    """Encapsulates Claude 3 model invocations using the Amazon Bedrock Runtime client."""

    def __init__(self, client=None, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        :param client: A low-level client representing Amazon Bedrock Runtime.
                       Describes the API operations for running inference using Bedrock models.
        """
        if client is None:
            client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        self._client = client
        self._model_id = model_id
        try:
            with open(CONFIG_PATH, "r") as f:
                self._config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading Bedrock config: {e}")
        self._pattern = fr'{self._config['regex']}'

    def infer(self, prompt: PromptTemplate, image: Image.Image, image_name: str) -> EnrichImageResult:
        """
        Generates a caption for the given image using the Bedrock Claude 3 Sonnet model.

        Args:
            prompt (PromptTemplate): The prompt template to use for the model inference.
            image (PIL.Image.Image): The image to generate a caption for.
            image_name (str): The name of the image.

        Returns:
            EnrichImageResult: The result of the model inference, including the generated caption.
        """
        content = [
            {
                "type": "text",
                "text": prompt.prompt_prefix,
            }
        ]
        if prompt.cot_prompt is not None:
            content.append(
                {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": self._encode_image_from_path(prompt.cot_prompt.image, target_width=128, target_height=128),
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": prompt.cot_prompt.explanation,
                }
            )
        content.append(
            {
                "type": "text",
                "text": prompt.prompt_suffix,
            }
        )
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": prompt.system_prompt,
            "temperature": 1,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
        }

        return self._generate_caption(image_name, request_body)

    @retry_exp_backoff(attempts=5, initial_delay=2, backoff_factor=2)
    def _generate_caption(self, image_name, request_body):
        """
        Generates a caption for the given image using the Bedrock Claude 3 Sonnet model.

        :param image_name: The name of the image.
        :param request_body: The request body to send to the Bedrock model.

        :return: The result of the model inference, including the generated caption.
        """
        try:
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=json.dumps(request_body),
            )
            llm_response = json.loads(response.get("body").read())
            answer = self._extract_caption(llm_response)
            return EnrichImageResult(image_name=image_name,
                                     llm_response=llm_response,
                                     caption=answer)
        except Exception as err:
            print(
                f"Couldn't invoke Claude 3 Sonnet. Here's why: {err}"
            )
            raise

    def _encode_image_from_path(self, image: Image, target_width, target_height):
        """
        Encodes a PIL Image object to a base64-encoded JPEG string.

        This function takes a PIL Image object and resizes it to the specified target width and height, if provided.
        If only one of the target dimensions is provided, the other dimension is calculated to maintain the aspect ratio.
        The resized image is then saved to a BytesIO buffer, encoded as a base64 string, and returned.

        :param image: The input PIL Image object to be encoded.
        :param target_width: The target width for the resized image. If not provided, the aspect ratio is maintained.
        :param target_height: The target height for the resized image. If not provided, the aspect ratio is maintained.

        :return: A base64-encoded JPEG string representation of the resized image.
        """
        if target_width is not None and target_height is not None:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        elif target_width is not None:
            w_percent = (target_width / float(image.size[0]))
            h_size = int((float(image.size[1]) * float(w_percent)))
            image = image.resize((target_width, h_size), Image.Resampling.LANCZOS)
        elif target_height is not None:
            h_percent = (target_height / float(image.size[1]))
            w_size = int((float(image.size[0]) * float(h_percent)))
            image = image.resize((w_size, target_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_base64

    def _extract_caption(self, llm_response):
        """
        Extracts the caption from the LLM response using a regular expression pattern.

        :param llm_response: The response from the Bedrock LLM model.

        :return: The extracted caption.
        """
        match = re.search(self._pattern, llm_response)
        if match:
            caption = match.group(1)
            return caption
        logger.info(f"Claude response does not match regex: {self._pattern}.\nResponse string:{llm_response}")
        raise
