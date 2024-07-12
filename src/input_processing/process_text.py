from config.constants import CONSTANTS

def tokenize_captions(examples, tokenizer, text_max_length):
    captions = [caption for caption in examples[CONSTANTS.IMAGE_PATH_COLUMN]]
    tokens = tokenizer(captions,
                       max_length=text_max_length,
                       padding="max_length",
                       truncation=True,
                       return_tensors="pt",
                       return_token_type_ids=False)
    examples['input_ids'] = tokens['input_ids']
    examples['attention_mask'] = tokens['attention_mask']
    return examples
