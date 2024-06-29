import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderModel

from input_processing.process_images import ImageTransformation


class ModelLoader:
    @classmethod
    def load_model(cls,
                   vision_encoder_name_or_path: str,
                   text_encoder_name_or_path: str,
                   tokenizer_name_or_path: str,
                   image_processor_name_or_path: str,
                   data_parallel: bool):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_model_name_or_path=vision_encoder_name_or_path,
            text_model_name_or_path=text_encoder_name_or_path
        )
        image_size = model.config.vision_config.image_size
        if torch.cuda.is_available():
            model.to("cuda")
            if data_parallel:
                model = torch.nn.DataParallel(model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        image_processor = AutoImageProcessor.from_pretrained(image_processor_name_or_path)
        image_transformations = ImageTransformation(
            image_size, image_processor.image_mean, image_processor.image_std
        )
        image_transformations = torch.jit.script(image_transformations)
        return model, tokenizer, image_processor, image_transformations


if __name__ == '__main__':
    a = ModelLoader.load_model(
        vision_encoder_name_or_path="openai/clip-vit-base-patch32",
        text_encoder_name_or_path="google-bert/bert-base-multilingual-cased",
        tokenizer_name_or_path="google-bert/bert-base-multilingual-cased",
        image_processor_name_or_path="openai/clip-vit-base-patch32",
        data_parallel=True
    )
    print(a)