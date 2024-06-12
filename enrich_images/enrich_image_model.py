from abc import ABC, abstractmethod
from typing import Protocol, TypeVar
from dataclasses import dataclass
from prompt_template import PromptTemplate
from PIL import Image
T = TypeVar("T", bound="EnrichImageModel")

@dataclass
class EnrichImageResult: #TODO: maybe move this to utils/helpers
    """
    Dataclass to hold the result of the EnrichImageModel.infer() method.
    """
    image_name: str
    llm_response: str
    caption: str

class EnrichImageModel(ABC):
    """
    Abstract base class for models that enriches images with a caption based on a prompt.
    """

    @abstractmethod
    def infer(self, prompt: PromptTemplate, image: Image.Image) -> EnrichImageResult:
        """
        Abstract method to enrich the given image based on the provided prompt.

        Args:
            prompt (PromptTemplate): The prompt template to use for enriching the image.
            image (PIL.Image.Image): The image to be enriched with a caption.

        Returns:
            a
        """
        raise NotImplementedError


