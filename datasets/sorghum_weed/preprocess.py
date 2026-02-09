from transformers import AutoImageProcessor

import config
from datasets.utils import process_and_save
from datasets.sorghum_weed.dataset import WeedDataset
from datasets.sorghum_weed import definitions as ds_config


def main():
    print('--- Starting Dataset Preparation ---')
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

    # Train
    train_ds = WeedDataset(ds_config.TRAIN_IMG_DIR, ds_config.TRAIN_JSON, processor)
    process_and_save(train_ds, output_dir='Train')

    # Validate
    val_ds = WeedDataset(ds_config.VAL_IMG_DIR, ds_config.VAL_JSON, processor)
    process_and_save(val_ds, output_dir='Validate')

    # Test
    test_ds = WeedDataset(ds_config.TEST_IMG_DIR, ds_config.TEST_JSON, processor)
    process_and_save(test_ds, output_dir='Test')

    print('\n--- Processing Complete ---')


if __name__ == '__main__':
    main()
