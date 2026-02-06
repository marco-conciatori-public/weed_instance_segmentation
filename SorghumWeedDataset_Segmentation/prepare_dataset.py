import os
import torch
from transformers import AutoImageProcessor

import config
from data_utils import WeedDataset


def process_and_save(dataset, dataset_name: str) -> None:
    """
    Iterates through the dataset and saves each item as a .pt file.
    """
    output_dir = os.path.join(config.PROCESSED_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nProcessing {dataset_name} set...')
    print(f'Saving to {output_dir}')

    total = len(dataset)
    for i in range(total):
        if (i + 1) % 10 == 0:
            print(f'\tProcessed {i + 1}/{total} images...', end='\r')

        item = dataset[i]
        file_name = item['file_name']

        # Create a safe file_name (replace extension with .pt)
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(output_dir, f'{base_name}.pt')

        # Save the dictionary containing tensors and metadata
        torch.save(item, save_path)

    print(f'\tProcessed {total}/{total} images.')


def main():
    print('--- Starting Dataset Preparation ---')

    # Initialize Processor
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

    # 1. Train Split
    train_dataset = WeedDataset(
        image_folder_path=config.TRAIN_IMG_DIR,
        annotation_file_path=config.TRAIN_JSON,
        processor=processor,
    )
    process_and_save(dataset=train_dataset, dataset_name='Train')

    # 2. Validation Split
    val_dataset = WeedDataset(
        image_folder_path=config.VAL_IMG_DIR,
        annotation_file_path=config.VAL_JSON,
        processor=processor,
    )
    process_and_save(dataset=val_dataset, dataset_name='Validate')

    # 3. Test Split
    test_dataset = WeedDataset(
        image_folder_path=config.TEST_IMG_DIR,
        annotation_file_path=config.TEST_JSON,
        processor=processor,
    )
    process_and_save(dataset=test_dataset, dataset_name='Test')

    print('\n--- Processing Complete ---')
    print(f'Data saved in: {config.PROCESSED_DIR}')


if __name__ == '__main__':
    main()
