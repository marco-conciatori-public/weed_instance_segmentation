import os
import cv2
import json
import torch
import warnings
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from torchmetrics.detection import MeanAveragePrecision

# Suppress specific warning about unused arguments in preprocessor config
# This occurs because the checkpoint config has keys not used by the current processor version
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*The following named arguments are not valid.*"
)

# Paths
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'Train/')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'Annotations/TrainSorghumWeed_json.json')
VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'Validate/')
VAL_JSON = os.path.join(DATASET_ROOT, 'Annotations/ValidateSorghumWeed_json.json')
TEST_IMG_DIR = os.path.join(DATASET_ROOT, 'Test/')
TEST_JSON = os.path.join(DATASET_ROOT, 'Annotations/TestSorghumWeed_json.json')
OUTPUT_DIR = 'models/mask2former_finetuned'

# Model Config
MODEL_CHECKPOINT = 'facebook/mask2former-swin-large-coco-instance'
BATCH_SIZE = 2  # Reduce to 1 if OOM
LEARNING_RATE = 5e-5
EPOCHS = 10
GRADIENT_ACCUMULATION = 2
MAX_INPUT_DIM = 1024  # Resize images larger than this to save VRAM
MAX_IMAGES = 25

# Class Mapping (Internal ID -> Name)
# Mask2Former uses background implicitly
ID2LABEL = {
    0: "Sorghum",
    1: "BLweed",
    2: "Grass"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class WeedDataset(Dataset):
    def __init__(self, image_folder_path: str, annotation_file_path: str, processor):
        self.image_folder = image_folder_path
        self.processor = processor

        # Load JSON data
        with open(annotation_file_path, 'r') as f:
            self.data = list(json.load(f).values())

        # Filter out images that don't exist or have no regions
        self.valid_entries = []
        valid_image_count = 0
        for entry in self.data:
            img_path = os.path.join(self.image_folder, entry['filename'])
            if os.path.exists(img_path) and len(entry.get('regions', [])) > 0:
                self.valid_entries.append(entry)
                valid_image_count += 1
                if (MAX_IMAGES is not None) and (valid_image_count >= MAX_IMAGES):
                    break

        print(f"Loaded {len(self.valid_entries)} valid images from {annotation_file_path}")

    def __len__(self):
        return len(self.valid_entries)

    def __getitem__(self, idx):
        entry = self.valid_entries[idx]
        image_path = os.path.join(self.image_folder, entry['filename'])

        # 1. Load Image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # print(f"Original image size: {width}x{height}")

        # resizing logic
        # 6000x4000 is too large. Resize to a max dimension (MAX_INPUT_DIM) to ensure the instance map
        # creation doesn't consume too much RAM/CPU and to fit in VRAM
        scale_factor = 1.0
        if max(width, height) > MAX_INPUT_DIM:
            scale_factor = MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            width, height = new_width, new_height  # Update dims for mask creation
            # print(f"Resized image to: {width}x{height}")

        target_size = (height, width)  # This is what we need for post-processing

        # 2. Process Annotations (VIA Polygons -> Instance Maps)
        # Mask2Former Processor expects:
        # - segmentation_maps: (H, W) array where pixels = instance_id (1, 2, 3...)
        # - instance_id_to_semantic_id: dict mapping {instance_id: class_id}

        instance_map = np.zeros((height, width), dtype=np.int32)
        instance_id_to_semantic_id = {}

        regions = entry.get('regions', [])
        current_instance_id = 1  # Start from 1, 0 is background

        for region in regions:
            shape_attr = region['shape_attributes']
            region_attr = region['region_attributes']

            if shape_attr['name'] != 'polygon':
                continue

            # Get class ID
            class_name = region_attr.get('classname', None)
            if class_name not in LABEL2ID:
                continue  # Skip unknown classes

            class_id = LABEL2ID[class_name]

            # Draw polygon on the instance map
            # NOTE: We must scale the polygon points if we resized the image
            all_x = [int(x * scale_factor) for x in shape_attr['all_points_x']]
            all_y = [int(y * scale_factor) for y in shape_attr['all_points_y']]

            points = np.array(list(zip(all_x, all_y)), dtype=np.int32)

            # Fill the polygon with the current_instance_id
            cv2.fillPoly(instance_map, [points], color=current_instance_id)

            # Record the mapping
            instance_id_to_semantic_id[current_instance_id] = class_id
            current_instance_id += 1

        # 3. Use Processor to create tensor inputs
        # on-the-fly preprocessing
        inputs = self.processor(
            images=[image],
            segmentation_maps=[instance_map],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
            ignore_index=0  # Background pixels will be ignored in loss computation
        )

        # Un-batch the output from processor (list of length 1)
        return {
            "pixel_values": inputs["pixel_values"][0],
            "mask_labels": inputs["mask_labels"][0],
            "class_labels": inputs["class_labels"][0],
            "target_size": target_size
        }


def collate_fn(batch):
    # Custom collator to handle list of tensors of variable sizes (for mask_labels/class_labels)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # mask_labels and class_labels are lists because number of instances varies per image
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    target_sizes = [item["target_size"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "target_sizes": target_sizes
    }


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


def test_with_metrics(model, processor, data_loader, device):
    """
    Evaluates the model on the test set using detailed metrics (mAP).
    Note: This requires the `torchmetrics` library to be installed (`pip install torchmetrics`).
    """
    model.eval()
    # The metric expects a list of predictions and a list of targets.
    # Each prediction is a dict with 'masks', 'scores', 'labels'.
    # Each target is a dict with 'masks', 'labels'.
    map_metric = MeanAveragePrecision(iou_type="segm")

    for batch in tqdm(data_loader, desc="Calculating Metrics"):
        pixel_values = batch["pixel_values"].to(device)
        target_sizes = batch["target_sizes"]  # List of (height, width) tuples

        # Prepare ground truth targets for the metric
        targets = []
        for i in range(len(pixel_values)):
            targets.append({
                "masks": batch["mask_labels"][i],
                "labels": batch["class_labels"][i],
            })

        # Perform inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Post-process outputs to get instance segmentation predictions
        # This returns a list of dicts, one for each image in the batch
        predictions = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5,  # Confidence threshold for predictions
            mask_threshold=0.5  # Threshold to binarize masks
        )

        # The metric expects predictions and targets to be on the same device.
        # The processor puts predictions on the model's device. Targets are on CPU.
        # We'll move predictions to CPU.
        cpu_predictions = []
        for pred in predictions:
            cpu_predictions.append({k: v.cpu() for k, v in pred.items()})

        map_metric.update(cpu_predictions, targets)

    # Compute and return the final metrics
    results = map_metric.compute()
    model.train()  # Set model back to training mode
    return results


def print_metrics(metrics, model_name="Model"):
    """Helper function to print metrics from torchmetrics."""
    print(f"\n--- {model_name} Metrics ---")
    if not metrics:
        print("No metrics calculated.")
        return

    # Main mAP metrics
    print(f"  mAP:              {metrics.get('map', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (IoU=0.50):   {metrics.get('map_50', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (IoU=0.75):   {metrics.get('map_75', torch.tensor(-1)).item():.4f}")
    print("-" * 25)
    # mAP for different area sizes
    print(f"  mAP (small):      {metrics.get('map_small', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (medium):     {metrics.get('map_medium', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (large):      {metrics.get('map_large', torch.tensor(-1)).item():.4f}")


def prepare_metrics_for_json(metrics):
    """Converts tensors in metrics dict to floats for JSON serialization."""
    if not metrics:
        return None
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


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
    processor = AutoImageProcessor.from_pretrained(final_model_path)

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
