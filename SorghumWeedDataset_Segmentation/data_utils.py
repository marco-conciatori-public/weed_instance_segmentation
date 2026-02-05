import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import MAX_IMAGES, MAX_INPUT_DIM, LABEL2ID


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

        print(f'Loaded {len(self.valid_entries)} valid images from {annotation_file_path}')

    def __len__(self):
        return len(self.valid_entries)

    def __getitem__(self, idx):
        entry = self.valid_entries[idx]
        image_path = os.path.join(self.image_folder, entry['filename'])

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        scale_factor = 1.0
        if max(width, height) > MAX_INPUT_DIM:
            scale_factor = MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            width, height = new_width, new_height

        target_size = (height, width)

        # Initialize with 255 (ignore_index)
        instance_map = np.full((height, width), 255, dtype=np.int32)
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

            # don't accidentally use 255 as an instance ID
            if current_instance_id == 255:
                current_instance_id += 1

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
            return_tensors='pt',
            ignore_index=255  # Standard ignore index
        )

        return {
            'pixel_values': inputs['pixel_values'][0],
            'mask_labels': inputs['mask_labels'][0],
            'class_labels': inputs['class_labels'][0],
            'target_size': target_size,
            'original_map': instance_map,  # Return raw map for accurate evaluation
            'id_to_semantic': instance_id_to_semantic_id
        }


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    mask_labels = [item['mask_labels'] for item in batch]
    class_labels = [item['class_labels'] for item in batch]
    target_sizes = [item['target_size'] for item in batch]

    # Collect new fields for evaluation
    original_maps = [item['original_map'] for item in batch]
    id_mappings = [item['id_to_semantic'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'mask_labels': mask_labels,
        'class_labels': class_labels,
        'target_sizes': target_sizes,
        'original_maps': original_maps,
        'id_mappings': id_mappings
    }
