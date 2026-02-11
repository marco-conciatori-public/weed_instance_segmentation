import os
import glob
import cv2
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class CropWeedDataset(Dataset):
    """
    Dataset loader for Crop Weed Field Image Dataset (CWFID) using YAML annotations.
    Expects:
    - Images folder containing RGB images.
    - Annotations folder containing .yaml files with polygon definitions.
    """
    def __init__(self, image_folder_path, annotation_file_path, processor, label2id: dict):
        self.image_folder = image_folder_path
        self.annotation_folder = annotation_file_path
        self.processor = processor
        self.label2id = label2id

        # List all YAML files in the annotation folder
        yaml_files = glob.glob(os.path.join(self.annotation_folder, '*.yaml'))
        yaml_files.sort()

        self.valid_files = []
        valid_count = 0

        print(f"Scanning {len(yaml_files)} annotation files in {self.annotation_folder}...")

        for yaml_path in yaml_files:
            try:
                # Read the YAML to find the corresponding image filename
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                img_filename = data.get('filename')
                if not img_filename:
                    continue

                # Construct image path
                img_path = os.path.join(self.image_folder, img_filename)

                if os.path.exists(img_path):
                    self.valid_files.append((img_path, yaml_path))
                    valid_count += 1

                    if (config.MAX_IMAGES is not None) and (valid_count >= config.MAX_IMAGES):
                        break
            except Exception as e:
                print(f"Warning: Error reading {yaml_path}: {e}")

        print(f'\tLoaded {len(self.valid_files)} valid image/yaml pairs from "{self.image_folder}"')

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> dict:
        image_path, yaml_path = self.valid_files[idx]
        file_name = os.path.basename(image_path)

        # Load Image
        image = Image.open(image_path).convert('RGB')

        # Load YAML Annotation
        with open(yaml_path, 'r') as f:
            annotation_data = yaml.safe_load(f)

        width, height = image.size

        # Calculate Scale Factor if resizing is needed
        scale_factor = 1.0
        if max(width, height) > config.MAX_INPUT_DIM:
            scale_factor = config.MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            width, height = new_width, new_height

        target_size = (height, width)

        # --- Create Instance Map ---
        # Initialize with 255 (ignore/background)
        instance_map = np.full((height, width), 255, dtype=np.int32)
        instance_id_to_semantic_id = {}
        current_instance_id = 1

        # The 'annotation' key holds the list of regions
        regions = annotation_data.get('annotation', [])
        if regions is None:
            regions = []

        for region in regions:
            # Region type: 'weed' or 'crop'
            type_name = region.get('type')

            # Map string label to ID using the config provided label2id
            # Example: 'weed' -> 1, 'crop' -> 0
            if type_name not in self.label2id:
                continue

            class_id = self.label2id[type_name]

            if current_instance_id == 255:
                current_instance_id += 1

            # Extract points
            points_dict = region.get('points', {})
            xs = points_dict.get('x', [])
            ys = points_dict.get('y', [])

            if len(xs) != len(ys) or len(xs) < 3:
                continue  # Skip invalid polygons

            # Scale points
            poly_points = []
            for x, y in zip(xs, ys):
                poly_points.append([int(x * scale_factor), int(y * scale_factor)])

            points_np = np.array(poly_points, dtype=np.int32)

            # Draw filled polygon for this instance
            cv2.fillPoly(instance_map, [points_np], color=current_instance_id)

            instance_id_to_semantic_id[current_instance_id] = class_id
            current_instance_id += 1

        # Use the processor
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
            'file_name': file_name
        }
