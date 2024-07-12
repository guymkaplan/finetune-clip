import json
from dataclasses import dataclass
from PIL import Image
from typing import Optional
from utils.utils import load_image
@dataclass
class CoTPrompt:
    image: Optional[Image.Image]
    explanation: str
@dataclass
class PromptTemplate:
    """
    Dataclass to hold the components of a prompt template.
    """
    system_prompt: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    regex: str = ".*"
    cot_prompt: Optional[CoTPrompt] = None

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)

        cot_prompt = None
        if "cot_prompt" in data:
            cot_prompt_data = data["cot_prompt"]
            image_path = cot_prompt_data.get("image_path")
            try:
                image = load_image(image_path)
            except Exception as e:
                print(f"Error loading image from path: {image_path}")
                raise e
            cot_prompt = CoTPrompt(
                image=image,
                explanation=cot_prompt_data.get("explanation", ""),
            )
        return cls(
            system_prompt=data.get("system_prompt", ""),
            prompt_prefix=data.get("prompt_prefix", ""),
            prompt_suffix=data.get("prompt_suffix", ""),
            cot_prompt=cot_prompt
        )