import os
import json
import torch
import warnings
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

from config import (
    TRAIN_IMG_DIR, TRAIN_JSON, VAL_IMG_DIR, VAL_JSON, TEST_IMG_DIR, TEST_JSON,
    OUTPUT_DIR, MODEL_CHECKPOINT, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    GRADIENT_ACCUMULATION, MAX_INPUT_DIM, MAX_IMAGES, ID2LABEL, LABEL2ID
)
from data_utils import WeedDataset, collate_fn
from evaluation_utils import (
    test_with_metrics,
    print_metrics,
    prepare_metrics_for_json
)

# Suppress specific warning about unused arguments in preprocessor config
# This occurs because the checkpoint config has keys not used by the current processor version
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*The following named arguments are not valid.*"
)


def evaluate(model, data_loader, device, desc="Evaluating"):
    """Evaluates the model on a given dataset and returns the average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]

            # Forward Pass
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    model.train()  # Set model back to training mode
    return avg_loss


def train(output_dir, metadata):
    train_start_time = datetime.now()
    metadata['training'] = {
        'start_time': train_start_time.isoformat()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Initialize Processor
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    # 2. Datasets & Loaders
    train_dataset = WeedDataset(TRAIN_IMG_DIR, TRAIN_JSON, processor)
    val_dataset = WeedDataset(VAL_IMG_DIR, VAL_JSON, processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. Initialize Model
    # ignore_mismatched_sizes=True because it is replacing the 80-class COCO head
    # with our 3-class head.
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    model.train()
    best_val_loss = float('inf')

    print("Starting Training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]

            # Forward Pass
            # The model automatically computes loss if labels are provided
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            loss = outputs.loss

            # Normalize loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * GRADIENT_ACCUMULATION
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Complete. Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Step ---
        if val_loader:
            avg_val_loss = evaluate(model, val_loader, device, desc="Validating")
            print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(output_dir, "best_model")
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"\nSaved new best model to {save_path} with validation loss: {best_val_loss:.4f}")

    print("Training Complete")

    # Save Final Model
    final_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

    train_end_time = datetime.now()
    train_duration = (train_end_time - train_start_time).total_seconds()
    metadata['training']['end_time'] = train_end_time.isoformat()
    metadata['training']['duration_seconds'] = train_duration
    metadata['training']['best_validation_loss'] = best_val_loss if best_val_loss != float('inf') else None

    return metadata


def main():
    # Create a unique output directory for this run based on the current time
    run_start_time = datetime.now()
    run_timestamp = run_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    print(f"Results for this run will be saved in: {run_output_dir}")
    os.makedirs(run_output_dir, exist_ok=True)

    metadata = {
        'run_id': run_timestamp,
        'run_start_time': run_start_time.isoformat(),
        'model_config': {
            'base_checkpoint': MODEL_CHECKPOINT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'gradient_accumulation': GRADIENT_ACCUMULATION,
            'max_input_dim': MAX_INPUT_DIM,
        },
        'dataset_config': {
            'train_annotations': TRAIN_JSON,
            'validation_annotations': VAL_JSON,
            'test_annotations': TEST_JSON,
            'max_images_per_split': MAX_IMAGES,
        },
        'training': {},
        'testing': {}
    }

    # Run training
    metadata = train(run_output_dir, metadata)

    # --- Testing Step ---
    print("\n--- Starting Final Testing ---")
    test_start_time = datetime.now()
    metadata['testing']['start_time'] = test_start_time.isoformat()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to the models saved during training
    final_model_path = os.path.join(run_output_dir, "final_model")
    best_model_path = os.path.join(run_output_dir, "best_model")

    # We need a processor to create the test dataset.
    # It's best to load the one saved with the model to ensure consistency.
    if not os.path.exists(final_model_path):
        print("Final model not found. Skipping testing.")
        return

    # Use the processor associated with the trained model
    processor = AutoImageProcessor.from_pretrained(final_model_path, use_fast=False)

    # Create test dataset and loader
    test_dataset = WeedDataset(TEST_IMG_DIR, TEST_JSON, processor)
    if len(test_dataset) == 0:
        print("No test data found. Skipping testing.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Test the 'final_model'
    print(f"\nTesting final model from: {final_model_path}")
    final_model = Mask2FormerForUniversalSegmentation.from_pretrained(final_model_path).to(device)
    final_model_metrics = test_with_metrics(final_model, processor, test_loader, device)
    metadata['testing']['final_model_metrics'] = prepare_metrics_for_json(final_model_metrics)
    print_metrics(final_model_metrics, "Final Model")

    # Test the 'best_model' if it exists
    if os.path.exists(best_model_path):
        print(f"\nTesting best model from: {best_model_path}")
        best_model = Mask2FormerForUniversalSegmentation.from_pretrained(best_model_path).to(device)
        best_model_metrics = test_with_metrics(best_model, processor, test_loader, device)
        metadata['testing']['best_model_metrics'] = prepare_metrics_for_json(best_model_metrics)
        print_metrics(best_model_metrics, "Best Model")
    else:
        metadata['testing']['best_model_metrics'] = None

    print("\n--- Testing Complete ---")

    test_end_time = datetime.now()
    test_duration = (test_end_time - test_start_time).total_seconds()
    metadata['testing']['end_time'] = test_end_time.isoformat()
    metadata['testing']['duration_seconds'] = test_duration

    run_end_time = datetime.now()
    run_duration = (run_end_time - run_start_time).total_seconds()
    metadata['run_end_time'] = run_end_time.isoformat()
    metadata['total_duration_seconds'] = run_duration

    # Save metadata
    metadata_path = os.path.join(run_output_dir, 'run_metadata.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"\nRun metadata saved to {metadata_path}")
    except Exception as e:
        print(f"\nError saving metadata to {metadata_path}: {e}")


if __name__ == "__main__":
    main()
