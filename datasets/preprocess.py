from transformers import AutoImageProcessor

import config
from datasets.factory import get_dataset_and_config
from datasets.dataset_utils import process_and_save


def main():
    for dataset_name in config.DATASET_LIST:
        print(f'=== Processing Dataset: {dataset_name} ===')
        processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)
        WeedDataset, ds_config = get_dataset_and_config(config.DATASET_LIST[0])
        preprocessed_dataset_path = ds_config.DATASET_ROOT + 'preprocessed/'

        # Train
        train_ds = WeedDataset(
            image_folder_path=ds_config.TRAIN_IMG_DIR,
            annotation_file_path=ds_config.TRAIN_JSON,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(train_ds, output_dir=preprocessed_dataset_path + 'Train/')

        # Validate
        val_ds = WeedDataset(
            image_folder_path=ds_config.VAL_IMG_DIR,
            annotation_file_path=ds_config.VAL_JSON,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(val_ds, output_dir=preprocessed_dataset_path + 'Validate/')

        # Test
        test_ds = WeedDataset(
            image_folder_path=ds_config.TEST_IMG_DIR,
            annotation_file_path=ds_config.TEST_JSON,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )
        process_and_save(test_ds, output_dir=preprocessed_dataset_path + 'Test/')

        print(f'\tFinished processing {dataset_name}')

    print('\n--- Processing Complete ---')


if __name__ == '__main__':
    main()
