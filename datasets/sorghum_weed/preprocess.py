from transformers import AutoImageProcessor

import config
from datasets.utils import process_and_save
from datasets.factory import get_dataset_config
from datasets.sorghum_weed.dataset import WeedDataset


def main():
    print('--- Starting Dataset Preparation ---')
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)
    ds_config = get_dataset_config('sorghum_weed')

    # Train
    train_ds = WeedDataset(
        image_folder_path=ds_config.TRAIN_IMG_DIR,
        annotation_file_path=ds_config.TRAIN_JSON,
        processor=processor,
        label2id=ds_config.LABEL2ID,
    )
    process_and_save(train_ds, output_dir='Train')

    # Validate
    val_ds = WeedDataset(
        image_folder_path=ds_config.VAL_IMG_DIR,
        annotation_file_path=ds_config.VAL_JSON,
        processor=processor,
        label2id=ds_config.LABEL2ID,
    )
    process_and_save(val_ds, output_dir='Validate')

    # Test
    test_ds = WeedDataset(
        image_folder_path=ds_config.TEST_IMG_DIR,
        annotation_file_path=ds_config.TEST_JSON,
        processor=processor,
        label2id=ds_config.LABEL2ID,
    )
    process_and_save(test_ds, output_dir='Test')

    print('\n--- Processing Complete ---')


if __name__ == '__main__':
    main()
