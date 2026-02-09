import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config


def load_model(model_id: str):
    model_path = config.MODELS_OUTPUT_DIR + model_id
    print(f'Loading {model_path}...')
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, processor, device
