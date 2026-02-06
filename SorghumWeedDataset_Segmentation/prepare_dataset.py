import os
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

from config import (
    TRAIN_IMG_DIR, TRAIN_JSON, VAL_IMG_DIR, VAL_JSON, TEST_IMG_DIR, TEST_JSON,
    MODEL_CHECKPOINT, PROCESSED_DIR, MAX_IMAGES
)
from data_utils import WeedDataset


# Override MAX_IMAGES to ensure we process everything, unless specifically debugging
# MAX_IMAGES = None  # Uncomment this to force full processing regardless of config.py

def process_and_save(dataset, dataset_name: str):
    """
    Iterates through the dataset and saves each item as a .pt file.
    """
    output_dir = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {dataset_name} set...")
    print(f"Saving to {output_dir}")

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        filename = item['file_name']

        # Create a safe filename (replace extension with .pt)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}.pt")

        # Save the dictionary containing tensors and metadata
        torch.save(item, save_path)


def main():
    print("--- Starting Dataset Preparation ---")

    # Initialize Processor
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    # 1. Train Split
    train_dataset = WeedDataset(TRAIN_IMG_DIR, TRAIN_JSON, processor)
    process_and_save(train_dataset, 'Train')

    # 2. Validation Split
    val_dataset = WeedDataset(VAL_IMG_DIR, VAL_JSON, processor)
    process_and_save(val_dataset, 'Validate')

    # 3. Test Split
    test_dataset = WeedDataset(TEST_IMG_DIR, TEST_JSON, processor)
    process_and_save(test_dataset, 'Test')

    print("\n--- Processing Complete ---")
    print(f"Data saved in: {PROCESSED_DIR}")


if __name__ == '__main__':
    main()
