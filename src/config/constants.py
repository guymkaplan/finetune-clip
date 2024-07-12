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
    DEFAULT_CONTAINER_IMAGE: str = field(init=False)
    ENRICH_IMAGES_DEFAULT_INSTANCE_TYPE: str = field(init=False)
    DEFAULT_DESTINATION_BASE_PATH: str = field(init=False)
    USER_CODE_PATH: str = field(init=False)
    DEFAULT_USER_DIR_PATH: str = field(init=False)

    def __post_init__(self):
        if self.config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_config_path = os.path.join(current_dir, 'constants.json')
            self.config_path = os.environ.get('CONFIG_PATH', default_config_path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)
        class ConfigurationError(Exception):
            pass
        self.LLM_RESPONSE_COLUMN = config.get('LLM_RESPONSE_COLUMN')
        if self.LLM_RESPONSE_COLUMN is None:
            raise ConfigurationError("LLM_RESPONSE_COLUMN is not configured")

        self.CAPTION_COLUMN = config.get('CAPTION_COLUMN')
        if self.CAPTION_COLUMN is None:
            raise ConfigurationError("CAPTION_COLUMN is not configured")

        self.IMAGE_PATH_COLUMN = config.get('IMAGE_PATH_COLUMN')
        if self.IMAGE_PATH_COLUMN is None:
            raise ConfigurationError("IMAGE_PATH_COLUMN is not configured")

        self.RANDOM_SEED = config.get('RANDOM_SEED')
        if self.RANDOM_SEED is None:
            raise ConfigurationError("RANDOM_SEED is not configured")

        self.TEXT_MAX_LENGTH = config.get('TEXT_MAX_LENGTH')
        if self.TEXT_MAX_LENGTH is None:
            raise ConfigurationError("TEXT_MAX_LENGTH is not configured")

        self.DEFAULT_IMAGE_SIZE = config.get('DEFAULT_IMAGE_SIZE')
        self.DEFAULT_CONTAINER_IMAGE = config.get('DEFAULT_CONTAINER_IMAGE')
        self.ENRICH_IMAGES_DEFAULT_INSTANCE_TYPE = config.get('ENRICH_IMAGES_DEFAULT_INSTANCE_TYPE')
        self.DEFAULT_DESTINATION_BASE_PATH = config.get('DEFAULT_DESTINATION_BASE_PATH')
        self.USER_CODE_PATH = config.get('USER_CODE_PATH')
        self.DEFAULT_USER_DIR_PATH = config.get('DEFAULT_USER_DIR_PATH')




CONSTANTS = Constants()