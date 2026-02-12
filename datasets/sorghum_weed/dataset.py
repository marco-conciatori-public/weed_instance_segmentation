import os
import cv2
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class SorghumWeedDataset(Dataset):
    """
    Standard PyTorch Dataset for loading raw images and JSON annotations.
    """
    def __init__(self, image_folder_path, annotation_path, processor, label2id: dict):
        self.image_folder = image_folder_path
        self.processor = processor
        self.label2id = label2id

        with open(annotation_path, 'r') as f:
            self.data = list(json.load(f).values())

        # Filter valid entries
        self.valid_entries = []
        valid_image_count = 0
        for entry in self.data:
            img_path = os.path.join(self.image_folder, entry['filename'])
            if os.path.exists(img_path) and len(entry.get('regions', [])) > 0:
                self.valid_entries.append(entry)
                valid_image_count += 1
                if (config.MAX_IMAGES is not None) and (valid_image_count >= config.MAX_IMAGES):
                    break

        print(f'\t\tLoaded {len(self.valid_entries)} valid images from "{annotation_path}"')

    def __len__(self):
        return len(self.valid_entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.valid_entries[idx]
        image_path = os.path.join(self.image_folder, entry['filename'])

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Resize if too large
        scale_factor = 1.0
        if max(width, height) > config.MAX_INPUT_DIM:
            scale_factor = config.MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            width, height = new_width, new_height

        target_size = (height, width)

        # Create Instance Map (255 = background/ignore)
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
            if class_name not in self.label2id:
                continue
            class_id = self.label2id[class_name]

            if current_instance_id == 255:
                current_instance_id += 1

            all_x = [int(x * scale_factor) for x in shape_attr['all_points_x']]
            all_y = [int(y * scale_factor) for y in shape_attr['all_points_y']]
            points = np.array(list(zip(all_x, all_y)), dtype=np.int32)

            cv2.fillPoly(instance_map, [points], color=current_instance_id)

            instance_id_to_semantic_id[current_instance_id] = class_id
            current_instance_id += 1

        # Use the processor (handles normalization, padding)
        inputs = self.processor(
            images=[image],
            segmentation_maps=[instance_map],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors='pt',
            ignore_index=255
        )

        return {
            'pixel_values': inputs['pixel_values'][0],
            'mask_labels': inputs['mask_labels'][0],
            'class_labels': inputs['class_labels'][0],
            'target_size': target_size,
            'original_map': instance_map,
            'id_to_semantic': instance_id_to_semantic_id,
            'file_name': entry['filename']
        }
