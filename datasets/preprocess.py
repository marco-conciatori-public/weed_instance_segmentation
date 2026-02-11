import os
from transformers import AutoImageProcessor

import config
from datasets.factory import get_dataset_and_config
from datasets.dataset_utils import process_and_save


def main():
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)
    for dataset_name in config.DATASET_LIST:
        print(f'=== Processing Dataset: {dataset_name} ===')
        WeedDataset, ds_config = get_dataset_and_config(dataset_name)

        # check if dataset already preproccessed
        try:
            os.makedirs(ds_config.PROCESSED_DIR, exist_ok=False)
        except OSError:
            print(f'\tDataset "{dataset_name}" already preprocessed, skipping...\n')
            continue

        # check if the dataset has a predefined train/val/test split
        if not getattr(ds_config, 'TRAIN_IMG_DIR', False):
            # if not, create one from the main image folder and annotation file
            train_val_test_split = ds_config.TRAIN_VAL_TEST_SPLIT
            pass

        # Train
        train_ds = WeedDataset(
            image_folder_path=ds_config.TRAIN_IMG_DIR,
            annotation_file_path=ds_config.TRAIN_ANNOTATIONS,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(train_ds, output_dir=ds_config.PROCESSED_DIR + 'Train/')

        # Validate
        val_ds = WeedDataset(
            image_folder_path=ds_config.VAL_IMG_DIR,
            annotation_file_path=ds_config.VAL_ANNOTATIONS,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(val_ds, output_dir=ds_config.PROCESSED_DIR + 'Validate/')

        # Test
        test_ds = WeedDataset(
            image_folder_path=ds_config.TEST_IMG_DIR,
            annotation_file_path=ds_config.TEST_ANNOTATIONS,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(test_ds, output_dir=ds_config.PROCESSED_DIR + 'Test/')

        print(f'\tFinished processing {dataset_name}\n')

    print('--- Processing Complete ---\n')


if __name__ == '__main__':
    main()
