import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.sorghum_weed.dataset import WeedDataset
from datasets.utils import PreprocessedDataset, collate_fn
from datasets.sorghum_weed import definitions as ds_config
from datasets.sorghum_weed.preprocess import process_and_save

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


def train(output_dir, metadata: dict) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=False)

    # Ensure Data is Processed
    train_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Train')
    val_proc_path = os.path.join(ds_config.PROCESSED_DIR, 'Validate')

    if not os.path.exists(train_proc_path) or len(os.listdir(train_proc_path)) == 0:
        print("Pre-processing Train data...")
        raw_train = WeedDataset(ds_config.TRAIN_IMG_DIR, ds_config.TRAIN_JSON, processor)
        process_and_save(raw_train, dataset_name='Train')

    if not os.path.exists(val_proc_path) or len(os.listdir(val_proc_path)) == 0:
        print("Pre-processing Validation data...")
        raw_val = WeedDataset(ds_config.VAL_IMG_DIR, ds_config.VAL_JSON, processor)
        process_and_save(raw_val, dataset_name='Validate')

    train_loader = DataLoader(
        PreprocessedDataset(train_proc_path),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        PreprocessedDataset(val_proc_path),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize Model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        config.MODEL_CHECKPOINT,
        id2label=ds_config.ID2LABEL,
        label2id=ds_config.LABEL2ID,
        ignore_mismatched_sizes=True
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
    run_output_dir = os.path.join(SPECIFIC_OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    metadata = {'run_id': timestamp}
    train(run_output_dir, metadata)


if __name__ == '__main__':
    main()
