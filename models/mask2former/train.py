import os
import json
import torch
import warnings
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.factory import get_dataset_and_config
from datasets.dataset_utils import PreprocessedDataset, collate_fn, process_and_save
from models.metrics import test_with_metrics, prepare_metrics_for_json, print_metrics_evaluation

warnings.filterwarnings('ignore', category=UserWarning, message='.*The following named arguments are not valid.*')
SPECIFIC_OUTPUT_DIR = config.MODELS_OUTPUT_DIR + 'mask2former_fine_tuned/'


def evaluate(model, data_loader, device, description: str = 'Evaluating') -> float:
    model.eval()
    total_loss = 0
    print(f'\tStarting {description}')
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            pixel_values = batch['pixel_values'].to(device)
            mask_labels = [labels.to(device) for labels in batch['mask_labels']]
            class_labels = [labels.to(device) for labels in batch['class_labels']]

            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            total_loss += outputs.loss.item()
            if (i + 1) % 10 == 0:
                print(f'\t\t{description} Step {i + 1}/{len(data_loader)} - Loss: {outputs.loss.item():.4f}')

    return total_loss / len(data_loader)


def get_unified_labels(dataset_list: list) -> tuple[dict, dict]:
    """
    Merges ID2LABEL maps from multiple datasets into a single unified mapping.
    Ensures no ID conflicts if classes differ.
    """
    unified_id2label = {}

    # Simple strategy: Merge dictionaries.
    # If IDs conflict, we assume the user maintains a consistent ID schema across datasets.
    # Otherwise, you would need to re-map IDs here.
    for ds_name in dataset_list:
        _, ds_config = get_dataset_and_config(ds_name)
        for id_num, label in ds_config.ID2LABEL.items():
            if id_num in unified_id2label and unified_id2label[id_num] != label:
                print(f'WARNING: ID collision for {id_num} ({unified_id2label[id_num]} vs {label}).'
                      f' Keeping {unified_id2label[id_num]}.')
            else:
                unified_id2label[id_num] = label

    unified_label2id = {v: k for k, v in unified_id2label.items()}
    print(f'Unified Classes: {unified_id2label}')
    return unified_id2label, unified_label2id


def format_duration(start_dt: datetime, end_dt: datetime) -> str:
    """Calculates duration between two datetime objects."""
    duration = end_dt - start_dt
    # Convert to string and remove microseconds
    return str(duration).split('.')[0]


def train(output_dir, metadata: dict, dataset_list: list) -> dict:
    try:
        start_time = datetime.now()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Training on: {device}')

        # 1. Prepare Unified Labels
        unified_id2label, unified_label2id = get_unified_labels(dataset_list)
        processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

        train_datasets = []
        val_datasets = []
        test_datasets = []

        # 2. Iterate and Load/Process each dataset
        for dataset_name in dataset_list:
            print(f'\n--- Preparing Dataset: {dataset_name} ---')
            WeedDataset, ds_config = get_dataset_and_config(dataset_name)

            train_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Train')
            val_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Validate')
            test_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Test')

            # Check if we need to pre-process (Train)
            if not os.path.exists(train_proc_path) or len(os.listdir(train_proc_path)) == 0 or config.FORCE_PREPROCESSING:
                print(f'\tPre-processing {dataset_name} Train data...')
                # Pass the UNIFIED label map so all datasets speak the same language
                raw_train = WeedDataset(
                    image_folder_path=ds_config.TRAIN_IMG_DIR,
                    annotation_path=ds_config.TRAIN_ANNOTATIONS,
                    processor=processor,
                    label2id=unified_label2id,
                )
                process_and_save(raw_train, output_dir=train_proc_path)

            # Check if we need to pre-process (Validate)
            if not os.path.exists(val_proc_path) or len(os.listdir(val_proc_path)) == 0 or config.FORCE_PREPROCESSING:
                print(f'\tPre-processing {dataset_name} Validation data...')
                raw_val = WeedDataset(
                    image_folder_path=ds_config.VAL_IMG_DIR,
                    annotation_path=ds_config.VAL_ANNOTATIONS,
                    processor=processor,
                    label2id=unified_label2id,
                )
                process_and_save(raw_val, output_dir=val_proc_path)

            # Check if we need to pre-process (Test)
            if not os.path.exists(test_proc_path) or len(os.listdir(test_proc_path)) == 0 or config.FORCE_PREPROCESSING:
                print(f'\tPre-processing {dataset_name} Test data...')
                raw_test = WeedDataset(
                    image_folder_path=ds_config.TEST_IMG_DIR,
                    annotation_path=ds_config.TEST_ANNOTATIONS,
                    processor=processor,
                    label2id=unified_label2id,
                )
                process_and_save(raw_test, output_dir=test_proc_path)

            train_datasets.append(PreprocessedDataset(train_proc_path))
            val_datasets.append(PreprocessedDataset(val_proc_path))
            test_datasets.append(PreprocessedDataset(test_proc_path))

        # 3. Concatenate Datasets
        full_train_dataset = ConcatDataset(train_datasets)
        full_val_dataset = ConcatDataset(val_datasets)
        full_test_dataset = ConcatDataset(test_datasets)

        print(f'\n\tCombined Training Samples: {len(full_train_dataset)}')
        print(f'\tCombined Validation Samples: {len(full_val_dataset)}')
        print(f'\tCombined Test Samples: {len(full_test_dataset)}')

        end_time = datetime.now()
        elapsed_time = format_duration(start_time, end_time)
        print(f'\tData preprocessing completed in {elapsed_time}')
        metadata['preprocessing_time'] = elapsed_time
        start_time = end_time

        train_loader = DataLoader(
            dataset=full_train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            dataset=full_val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            dataset=full_test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Initialize Model with UNIFIED configuration
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.MODEL_CHECKPOINT,
            id2label=unified_id2label,
            label2id=unified_label2id,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        best_val_loss = float('inf')
        metadata['training_history'] = []
        model.train()
        print('Starting Training...')

        end_time = datetime.now()
        elapsed_time = format_duration(start_time, end_time)
        print(f'\tData and model loading completed in {elapsed_time}')
        metadata['data_and_model_loading_time'] = elapsed_time
        start_time = end_time

        for epoch in range(config.EPOCHS):
            total_loss = 0
            print(f'\nEpoch {epoch + 1}/{config.EPOCHS}')

            for step, batch in enumerate(train_loader):
                pixel_values = batch['pixel_values'].to(device)
                mask_labels = [ml.to(device) for ml in batch['mask_labels']]
                class_labels = [cl.to(device) for cl in batch['class_labels']]

                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss / config.GRADIENT_ACCUMULATION
                loss.backward()

                if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                current_loss = loss.item() * config.GRADIENT_ACCUMULATION
                total_loss += current_loss
                # print(f'\t\tStep {step + 1}/{len(train_loader)} - Loss: {current_loss:.4f}')

            avg_train_loss = total_loss / len(train_loader)
            print(f'\tEpoch {epoch + 1} Avg Loss: {avg_train_loss:.4f}')

            avg_val_loss = evaluate(model=model, data_loader=val_loader, device=device, description='Validation')
            print(f'\tEpoch {epoch + 1} Val Loss: {avg_val_loss:.4f}')

            # Log epoch stats
            metadata['training_history'].append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(output_dir, 'best_model')
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f'\tSaved new best model (Loss: {best_val_loss:.4f})')

        end_time = datetime.now()
        elapsed_time = format_duration(start_time, end_time)
        print(f'\tTraining completed in {elapsed_time}')
        metadata['training_time'] = elapsed_time

        final_path = os.path.join(output_dir, 'final_model')
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)

        # --- Test Phase ---
        print('\n--- Starting Test Phase (Best Model) ---')
        best_model_path = os.path.join(output_dir, 'best_model')

        start_time = datetime.now()

        if os.path.exists(best_model_path):
            print(f'\tLoading best model from {best_model_path}')
            best_model = Mask2FormerForUniversalSegmentation.from_pretrained(best_model_path).to(device)
            best_processor = AutoImageProcessor.from_pretrained(best_model_path, use_fast=False)

            test_results = test_with_metrics(
                model=best_model,
                processor=best_processor,
                data_loader=test_loader,
                device=device,
            )
            print_metrics_evaluation(metrics_evaluation=test_results, model_name='Best Model')

            # Add to metadata
            clean_metrics = prepare_metrics_for_json(test_results)
            metadata['test_metrics'] = clean_metrics

        else:
            print('\tBest model not found, skipping test phase.')

        end_time = datetime.now()
        elapsed_time = format_duration(start_time, end_time)
        print(f'\tTest completed in {elapsed_time}')
        metadata['test_time'] = elapsed_time

        return metadata

    except Exception as e:
        print(f'\nError during training/testing:\n\t{e}')
        return metadata


def main():
    global_start_time = datetime.now()
    run_output_dir = os.path.join(SPECIFIC_OUTPUT_DIR, f'{global_start_time.strftime('%Y-%m-%d_%H-%M-%S')}')
    os.makedirs(run_output_dir, exist_ok=True)

    metadata = {
        'start_time': global_start_time.strftime('%Y-%m-%d_%H-%M-%S'),
        'dataset_list': config.DATASET_LIST,
        'base_model': config.MODEL_CHECKPOINT,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.EPOCHS,
        'gradient_accumulation': config.GRADIENT_ACCUMULATION,
        'max_input_dim': config.MAX_INPUT_DIM,
    }
    # Save metadata at the beginning in case of a crash for long training runs
    metadata_path = os.path.join(run_output_dir, 'metadata.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f'\nError in saving metadata to "{metadata_path}":\n\t {e}')

    # Metadata is updated in place during train()
    updated_metadata = train(output_dir=run_output_dir, metadata=metadata, dataset_list=config.DATASET_LIST)
    global_end_time = datetime.now()
    updated_metadata['end_time'] = global_end_time.strftime('%Y-%m-%d_%H-%M-%S')
    updated_metadata['total_time'] = format_duration(global_start_time, global_end_time)

    # Update metadata
    try:
        with open(metadata_path, 'w') as f:
            json.dump(updated_metadata, f, indent=4)
    except Exception as e:
        print(f'\nError in updating metadata to "{metadata_path}":\n\t {e}')


if __name__ == '__main__':
    main()
