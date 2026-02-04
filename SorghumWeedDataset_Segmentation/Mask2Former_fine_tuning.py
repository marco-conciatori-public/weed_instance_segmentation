import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

# Paths
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'Train/')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'Annotations/TrainSorghumWeed_json.json')
VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'Validation/')
VAL_JSON = os.path.join(DATASET_ROOT, 'Annotations/ValidationSorghumWeed_json.json')
OUTPUT_DIR = 'SorghumWeedDataset_Segmentation/models/mask2former_finetuned'

# Model Config
MODEL_CHECKPOINT = 'facebook/mask2former-swin-large-coco-instance'
BATCH_SIZE = 2  # Reduce to 1 if OOM
LEARNING_RATE = 5e-5
EPOCHS = 10
GRADIENT_ACCUMULATION = 2

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
        for entry in self.data:
            img_path = os.path.join(self.image_folder, entry['filename'])
            if os.path.exists(img_path) and len(entry.get('regions', [])) > 0:
                self.valid_entries.append(entry)

        print(f"Loaded {len(self.valid_entries)} valid images from {annotation_file_path}")

    def __len__(self):
        return len(self.valid_entries)

    def __getitem__(self, idx):
        entry = self.valid_entries[idx]
        image_path = os.path.join(self.image_folder, entry['filename'])

        # 1. Load Image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

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
            all_x = shape_attr['all_points_x']
            all_y = shape_attr['all_points_y']
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
            task_inputs=["instance"],
            return_tensors="pt"
        )

        # Un-batch the output from processor (list of length 1)
        return {
            "pixel_values": inputs["pixel_values"][0],
            "mask_labels": inputs["mask_labels"][0],
            "class_labels": inputs["class_labels"][0]
        }


def collate_fn(batch):
    # Custom collator to handle list of tensors of variable sizes (for mask_labels/class_labels)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # mask_labels and class_labels are lists because number of instances varies per image
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }


def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Processor
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    # 2. Datasets & Loaders
    train_dataset = WeedDataset(TRAIN_IMG_DIR, TRAIN_JSON, processor)
    # Using Train set for validation just to check code logic, ideally use Val set
    val_dataset = WeedDataset(VAL_IMG_DIR, VAL_JSON, processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # Verify validation loader works
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            # Save Checkpoint
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

    print("Training Complete")

    # Save Final Model
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    training()
