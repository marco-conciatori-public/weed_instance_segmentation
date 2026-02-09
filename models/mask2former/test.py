import os
import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.utils import collate_fn
from datasets.sorghum_weed.dataset import WeedDataset
from datasets.sorghum_weed import definitions as ds_config
from models.metrics import test_with_metrics, print_metrics_evaluation

MODEL_ID = 'mask2former_fine_tuned/2026-02-09_19-50-56/best_model/'


def test_model(model_id: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config.MODELS_OUTPUT_DIR + model_id

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)

    print("Loading Test Dataset...")
    test_ds = WeedDataset(ds_config.TEST_IMG_DIR, ds_config.TEST_JSON, processor)
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    metrics = test_with_metrics(model, processor, loader, device)
    print_metrics_evaluation(metrics, model_name="Best Model")


if __name__ == '__main__':
    # Adjust this path to your specific run directory
    test_model(MODEL_ID)
