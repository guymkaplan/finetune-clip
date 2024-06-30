

from dataclasses import dataclass, field
import json
import os
from typing import Tuple


@dataclass
class Constants:
    config_path: str = field(default=None)
    LLM_RESPONSE_COLUMN: str = field(init=False)
    CAPTION_COLUMN: str = field(init=False)
    IMAGE_PATH_COLUMN: str = field(init=False)
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = field(init=False)
    RANDOM_SEED: int = field(init=False)
    TEXT_MAX_LENGTH: int = field(init=False)
    def __post_init__(self):
        if self.config_path is None:
            self.config_path = os.environ.get('CONFIG_PATH', 'constants.json')

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)

        self.LLM_RESPONSE_COLUMN = config.get('LLM_RESPONSE_COLUMN', '')
        self.CAPTION_COLUMN = config.get('CAPTION_COLUMN', '')
        self.IMAGE_PATH_COLUMN = config.get('IMAGE_PATH_COLUMN', '')
        self.DEFAULT_IMAGE_SIZE = tuple(config.get('DEFAULT_IMAGE_SIZE', (244, 244)))
        self.RANDOM_SEED = config.get('RANDOM_SEED', 42)
        self.TEXT_MAX_LENGTH = config.get('TEXT_MAX_LENGTH', 128)


CONSTANTS = Constants()