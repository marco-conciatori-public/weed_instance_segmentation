import os
import torch
from transformers import AutoImageProcessor

import config
from datasets.sorghum_weed.dataset import WeedDataset
from datasets.sorghum_weed import definitions as ds_config


def process_and_save(dataset, dataset_name: str) -> None:
    output_dir = os.path.join(ds_config.PROCESSED_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nProcessing {dataset_name} set...')
    print(f'Saving to {output_dir}')

    total = len(dataset)
    for i in range(total):
        if (i + 1) % 10 == 0:
            print(f'\tProcessed {i + 1}/{total} images...', end='\r')

        item = dataset[i]
        file_name = item['file_name']
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(output_dir, f'{base_name}.pt')
        torch.save(item, save_path)

    print(f'\tProcessed {total}/{total} images.')


def main():
    print('--- Starting Dataset Preparation ---')
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

    # Train
    train_ds = WeedDataset(ds_config.TRAIN_IMG_DIR, ds_config.TRAIN_JSON, processor)
    process_and_save(train_ds, dataset_name='Train')

    # Validate
    val_ds = WeedDataset(ds_config.VAL_IMG_DIR, ds_config.VAL_JSON, processor)
    process_and_save(val_ds, dataset_name='Validate')

    # Test
    test_ds = WeedDataset(ds_config.TEST_IMG_DIR, ds_config.TEST_JSON, processor)
    process_and_save(test_ds, dataset_name='Test')

    print('\n--- Processing Complete ---')


if __name__ == '__main__':
    main()
