import os
import cv2
import json
import torch
import warnings
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

# Suppress specific warning about unused arguments in preprocessor config
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=".*The following named arguments are not valid.*"
)

# Paths (These should point to your test data)
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'
TEST_IMG_DIR = os.path.join(DATASET_ROOT, 'Test/')
TEST_JSON = os.path.join(DATASET_ROOT, 'Annotations/TestSorghumWeed_json.json')

# Config (Adjust if needed for testing)
BATCH_SIZE = 2
MAX_INPUT_DIM = 1024
MAX_IMAGES = None  # Set to None to test on the full dataset, or a number to limit it

# Class Mapping
ID2LABEL = {
    0: "Sorghum",
    1: "BLweed",
    2: "Grass"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class WeedDataset(Dataset):
    """
    Dataset class for loading weed segmentation data.
    This is a copy of the class from the training script.
    """
    def __init__(self, image_folder_path: str, annotation_file_path: str, processor):
        self.image_folder = image_folder_path
        self.processor = processor

        with open(annotation_file_path, 'r') as f:
            self.data = list(json.load(f).values())

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
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        scale_factor = 1.0
        if max(width, height) > MAX_INPUT_DIM:
            scale_factor = MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            width, height = new_width, new_height

        target_size = (height, width)
        instance_map = np.zeros((height, width), dtype=np.int32)
        instance_id_to_semantic_id = {}
        regions = entry.get('regions', [])
        current_instance_id = 1

        for region in regions:
            shape_attr = region['shape_attributes']
            region_attr = region['region_attributes']
            if shape_attr['name'] != 'polygon':
                continue
            class_name = region_attr.get('classname', None)
            if class_name not in LABEL2ID:
                continue
            class_id = LABEL2ID[class_name]
            all_x = [int(x * scale_factor) for x in shape_attr['all_points_x']]
            all_y = [int(y * scale_factor) for y in shape_attr['all_points_y']]
            points = np.array(list(zip(all_x, all_y)), dtype=np.int32)
            cv2.fillPoly(instance_map, [points], color=current_instance_id)
            instance_id_to_semantic_id[current_instance_id] = class_id
            current_instance_id += 1

        inputs = self.processor(
            images=[image],
            segmentation_maps=[instance_map],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
            ignore_index=0
        )

        return {
            "pixel_values": inputs["pixel_values"][0],
            "mask_labels": inputs["mask_labels"][0],
            "class_labels": inputs["class_labels"][0],
            "target_size": target_size
        }


def collate_fn(batch):
    """Custom collator from the training script."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    target_sizes = [item["target_size"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "target_sizes": target_sizes
    }


def test_with_metrics(model, processor, data_loader, device):
    """Evaluates the model on the test set using mAP metrics."""
    model.eval()
    map_metric = MeanAveragePrecision(iou_type="segm")

    for batch in tqdm(data_loader, desc="Calculating Metrics"):
        pixel_values = batch["pixel_values"].to(device)
        target_sizes = batch["target_sizes"]

        targets = []
        for i in range(len(pixel_values)):
            targets.append({
                "masks": batch["mask_labels"][i].to(torch.bool),
                "labels": batch["class_labels"][i],
            })

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        predictions = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5,
            mask_threshold=0.5
        )

        formatted_predictions = []
        for pred in predictions:
            segments_info = pred['segments_info']
            if not segments_info:
                formatted_predictions.append({
                    'masks': torch.empty(0, *pred['segmentation'].shape, dtype=torch.bool, device='cpu'),
                    'scores': torch.empty(0, device='cpu'),
                    'labels': torch.empty(0, dtype=torch.long, device='cpu'),
                })
                continue

            scores = torch.tensor([info['score'] for info in segments_info])
            labels = torch.tensor([info['label_id'] for info in segments_info])
            instance_map = pred['segmentation']
            instance_ids = [info['id'] for info in segments_info]
            masks = torch.stack([instance_map == iid for iid in instance_ids])

            formatted_predictions.append({
                'masks': masks.cpu(),
                'scores': scores.cpu(),
                'labels': labels.cpu(),
            })

        map_metric.update(formatted_predictions, targets)

    results = map_metric.compute()
    return results


def print_metrics(metrics, model_name="Model"):
    """Helper function to print metrics from torchmetrics."""
    print(f"\n--- {model_name} Metrics ---")
    if not metrics:
        print("No metrics calculated.")
        return
    print(f"  mAP:              {metrics.get('map', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (IoU=0.50):   {metrics.get('map_50', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (IoU=0.75):   {metrics.get('map_75', torch.tensor(-1)).item():.4f}")
    print("-" * 25)
    print(f"  mAP (small):      {metrics.get('map_small', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (medium):     {metrics.get('map_medium', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (large):      {metrics.get('map_large', torch.tensor(-1)).item():.4f}")


def main(args):
    """Main function to run the testing."""
    run_dir = args.model_run_dir
    if not os.path.isdir(run_dir):
        print(f"Error: Directory not found at '{run_dir}'")
        return

    print(f"--- Starting Testing for run: {os.path.basename(run_dir)} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    final_model_path = os.path.join(run_dir, "final_model")
    best_model_path = os.path.join(run_dir, "best_model")

    processor_path = final_model_path if os.path.exists(final_model_path) else best_model_path
    if not os.path.exists(processor_path):
        print(f"Error: No model or processor found in '{run_dir}'. Searched for 'final_model' and 'best_model'.")
        return

    print(f"Loading processor from: {processor_path}")
    processor = AutoImageProcessor.from_pretrained(processor_path, use_fast=False)

    print("\nLoading test dataset...")
    test_dataset = WeedDataset(TEST_IMG_DIR, TEST_JSON, processor)
    if len(test_dataset) == 0:
        print("No test data found. Skipping testing.")
        return
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if os.path.exists(final_model_path):
        print(f"\nTesting final model from: {final_model_path}")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(final_model_path).to(device)
        metrics = test_with_metrics(model, processor, test_loader, device)
        print_metrics(metrics, "Final Model")
    else:
        print("\n'final_model' not found. Skipping.")

    if os.path.exists(best_model_path):
        print(f"\nTesting best model from: {best_model_path}")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(best_model_path).to(device)
        metrics = test_with_metrics(model, processor, test_loader, device)
        print_metrics(metrics, "Best Model")
    else:
        print("\n'best_model' not found. Skipping.")

    print("\n--- Testing Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned Mask2Former model.")
    parser.add_argument(
        "model_run_dir",
        type=str,
        help="Path to the specific training run directory (e.g., 'models/mask2former_finetuned/YYYY-MM-DD_HH-MM-SS')."
    )
    cli_args = parser.parse_args()
    main(cli_args)
