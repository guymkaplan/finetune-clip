import logging
from typing import Dict, Type
from enrich_image_model import EnrichImageModel



MODELS: Dict[str, Type['EnrichImageModel']] = {}
def register_model(name: str):
    """
    Decorator to register a model class with a given name.
    """

    def decorator(cls):
        MODELS[name] = cls
        return cls

    return decorator


class ModelNotSupportedError(Exception):
    pass


def build_model(model_name: str, **kwargs) -> 'EnrichImageModel':
    """
    Factory method to build an instance of a registered model.
    Args:
        model_name (str): The name of the model to instantiate.
        **kwargs: Additional arguments to pass to the model constructor.
    Returns:
        An instance of the requested model.
    Raises:
        ModelNotSupportedError: If the model is not registered.
    """
    try:
        model_cls = MODELS[model_name]
        return model_cls(**kwargs)
    except KeyError as e:
        logging.error(f"Model {model_name} is not supported", exc_info=True)
        raise ModelNotSupportedError(f"Model {model_name} is not supported") from e
