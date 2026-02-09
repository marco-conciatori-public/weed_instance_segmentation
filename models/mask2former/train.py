import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.factory import get_dataset_config
from datasets.sorghum_weed.dataset import WeedDataset
from datasets.utils import PreprocessedDataset, collate_fn, process_and_save

warnings.filterwarnings('ignore', category=UserWarning, message='.*The following named arguments are not valid.*')
SPECIFIC_OUTPUT_DIR = config.MODELS_OUTPUT_DIR + 'mask2former_fine_tuned/'


def evaluate(model, data_loader, device, desc: str = 'Evaluating') -> float:
    model.eval()
    total_loss = 0
    print(f'Starting: {desc}')
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
                print(f'  {desc} Step {i + 1}/{len(data_loader)} - Loss: {outputs.loss.item():.4f}')

    return total_loss / len(data_loader)


def get_unified_labels(dataset_list: list):
    """
    Merges ID2LABEL maps from multiple datasets into a single unified mapping.
    Ensures no ID conflicts if classes differ.
    """
    unified_id2label = {}

    # Simple strategy: Merge dictionaries.
    # If IDs conflict, we assume the user maintains a consistent ID schema across datasets.
    # Otherwise, you would need to re-map IDs here.
    for ds_name in dataset_list:
        ds_config = get_dataset_config(ds_name)
        for id_num, label in ds_config.ID2LABEL.items():
            if id_num in unified_id2label and unified_id2label[id_num] != label:
                print(f"WARNING: ID collision for {id_num} ({unified_id2label[id_num]} vs {label}). Keeping {unified_id2label[id_num]}.")
            else:
                unified_id2label[id_num] = label

    unified_label2id = {v: k for k, v in unified_id2label.items()}
    print(f"Unified Classes: {unified_id2label}")
    return unified_id2label, unified_label2id


def train(output_dir, metadata: dict, dataset_list: list) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    # 1. Prepare Unified Labels
    unified_id2label, unified_label2id = get_unified_labels(dataset_list)
    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

    train_datasets = []
    val_datasets = []

    # 2. Iterate and Load/Process each dataset
    for dataset_name in dataset_list:
        print(f"\n--- Preparing Dataset: {dataset_name} ---")
        ds_config = get_dataset_config(dataset_name)

        train_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Train')
        val_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Validate')

        # Check if we need to pre-process (Train)
        # Note: If you change label mappings often, you might need to force re-processing here.
        if not os.path.exists(train_proc_path) or len(os.listdir(train_proc_path)) == 0:
            print(f"Pre-processing {dataset_name} Train data...")
            # Pass the UNIFIED label map so all datasets speak the same language
            raw_train = WeedDataset(ds_config.TRAIN_IMG_DIR, ds_config.TRAIN_JSON, processor, label2id=unified_label2id)
            process_and_save(raw_train, output_dir=train_proc_path)

        # Check if we need to pre-process (Validate)
        if not os.path.exists(val_proc_path) or len(os.listdir(val_proc_path)) == 0:
            print(f"Pre-processing {dataset_name} Validation data...")
            raw_val = WeedDataset(ds_config.VAL_IMG_DIR, ds_config.VAL_JSON, processor, label2id=unified_label2id)
            process_and_save(raw_val, output_dir=val_proc_path)

        train_datasets.append(PreprocessedDataset(train_proc_path))
        val_datasets.append(PreprocessedDataset(val_proc_path))

    # 3. Concatenate Datasets
    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)

    print(f"\nCombined Training Samples: {len(full_train_dataset)}")
    print(f"Combined Validation Samples: {len(full_val_dataset)}")

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        full_val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize Model with UNIFIED configuration
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        config.MODEL_CHECKPOINT,
        id2label=unified_id2label,
        label2id=unified_label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    model.train()
    print('Starting Training...')

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
            print(f'  Step {step + 1}/{len(train_loader)} - Loss: {current_loss:.4f}')

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} Avg Loss: {avg_train_loss:.4f}')

        avg_val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, 'best_model')
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f'Saved new best model (Loss: {best_val_loss:.4f})')

    final_path = os.path.join(output_dir, 'final_model')
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    return metadata


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_output_dir = os.path.join(SPECIFIC_OUTPUT_DIR, f"{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    metadata = {
        'run_id': timestamp,
        'dataset_list': config.DATASET_LIST,
    }
    train(run_output_dir, metadata, dataset_list=config.DATASET_LIST)


if __name__ == '__main__':
    main()
