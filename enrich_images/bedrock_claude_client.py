import base64
import json
import logging
from abc import ABC
from io import BytesIO

from PIL import Image
from prompt_template import PromptTemplate
from enrich_image_model import EnrichImageModel, EnrichImageResult
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockClaudeSonnet(EnrichImageModel):
    """Encapsulates Claude 3 model invocations using the Amazon Bedrock Runtime client."""

    def __init__(self, client=None, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        :param client: A low-level client representing Amazon Bedrock Runtime.
                       Describes the API operations for running inference using Bedrock models.
                       Default: None
        """
        if client is None:
            client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        self.client = client
        self.model_id = model_id

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    def infer(self, prompt: PromptTemplate, image: Image.Image) -> EnrichImageResult:
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
        #
        # ,)
        #         {
        #             "type": "image",
        #             "source": {
        #                 "type": "base64",
        #                 "media_type": "image/png",
        #                 "data": self._encode_image_from_path(image, target_width=128, target_height=128),
        #             },
        #         },
        #     {
        #     "type": "text",
        #     "text": f"{prompt_suffix}\n{task}",
        # })
        #
        # for i in range(len(task_base64_images)):
        #     content.append(
        #         {
        #             "type": "text",
        #             "text": f"{i + 1}) ",
        #         }
        #     )
        #     content.append(
        #         {
        #             "type": "image",
        #             "source": {
        #                 "type": "base64",
        #                 "media_type": "image/png",
        #                 "data": task_base64_images[i],
        #             },
        #         }
        #     )
        # content.append({
        #     "type": "text",
        #     "text": f"\nAnswer:",
        # })
        # # Invoke the model with the prompt and the encoded image
        # request_body = {
        #     "anthropic_version": "bedrock-2023-05-31",
        #     "max_tokens": 2048,
        #     "system": system_prompt,
        #     "temperature": 1,
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": content
        #         }
        #     ],
        # }

        try:
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
            )

            # Process and print the response
            result = json.loads(response.get("body").read())

            return result, content
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

        Args:
            image (PIL.Image.Image): The input PIL Image object to be encoded.
            target_width (int, optional): The target width for the resized image. If not provided, the aspect ratio is maintained.
            target_height (int, optional): The target height for the resized image. If not provided, the aspect ratio is maintained.

        Returns:
            str: A base64-encoded JPEG string representation of the resized image.
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
