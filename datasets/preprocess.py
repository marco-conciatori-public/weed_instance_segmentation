import os
import torch
from torch.utils.data import random_split
from transformers import AutoImageProcessor

import config
from datasets.factory import get_dataset_and_config
from datasets.dataset_utils import process_and_save


def main():
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)
    for dataset_name in config.DATASET_LIST:
        print(f'=== Processing Dataset: {dataset_name} ===')
        WeedDataset, ds_config = get_dataset_and_config(dataset_name)

        # check if dataset has already been preprocessed
        try:
            os.makedirs(ds_config.PROCESSED_DIR, exist_ok=False)
        except OSError:
            print(f'\tDataset "{dataset_name}" already preprocessed, skipping...\n')
            continue

        # Check if the dataset has a predefined train/val/test split
        try:
            split_ratios = getattr(ds_config, 'TRAIN_VAL_TEST_SPLIT')

            # Dynamic Split
            print(f'\tNo predefined split found. Splitting dataset with ratios {split_ratios}...')

            # Load the full dataset
            full_ds = WeedDataset(
                image_folder_path=ds_config.IMG_DIR,
                annotation_file_path=ds_config.ANNOTATIONS,
                processor=processor,
                label2id=ds_config.LABEL2ID,
            )

            total_size = len(full_ds)
            train_ratio, val_ratio, test_ratio = split_ratios

            # Calculate lengths ensuring they sum to total_size
            train_len = int(train_ratio * total_size)
            val_len = int(val_ratio * total_size)
            test_len = 0
            # Assign remainder to the last requested set to ensure sum matches total
            if test_ratio > 0:
                test_len = total_size - train_len - val_len

            print(f'\tSplit sizes: Train={train_len}, Val={val_len}, Test={test_len}')

            # Perform random split using a fixed generator for reproducibility
            train_subset, val_subset, test_subset = random_split(
                full_ds,
                [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42)
            )

            # Save Subsets (only if they have data)
            if train_len > 0:
                process_and_save(train_subset, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Train'))

            if val_len > 0:
                process_and_save(val_subset, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Validate'))

            if test_len > 0:
                process_and_save(test_subset, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Test'))

        except AttributeError:
            print(f'\tUsing predefined splits from {dataset_name} definitions.')

            # Train
            train_ds = WeedDataset(
                image_folder_path=ds_config.TRAIN_IMG_DIR,
                annotation_file_path=ds_config.TRAIN_ANNOTATIONS,
                processor=processor,
                label2id=ds_config.LABEL2ID,
            )
            process_and_save(train_ds, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Train'))

            # Validate
            val_ds = WeedDataset(
                image_folder_path=ds_config.VAL_IMG_DIR,
                annotation_file_path=ds_config.VAL_ANNOTATIONS,
                processor=processor,
                label2id=ds_config.LABEL2ID,
            )
            process_and_save(val_ds, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Validate'))

            # Test
            test_ds = WeedDataset(
                image_folder_path=ds_config.TEST_IMG_DIR,
                annotation_file_path=ds_config.TEST_ANNOTATIONS,
                processor=processor,
                label2id=ds_config.LABEL2ID,
            )
            process_and_save(test_ds, output_dir=os.path.join(ds_config.PROCESSED_DIR, 'Test'))

        print(f'\tFinished processing {dataset_name}\n')
    print('--- Processing Complete ---\n')


if __name__ == '__main__':
    main()
