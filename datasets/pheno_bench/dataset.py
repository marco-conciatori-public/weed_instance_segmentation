import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class PhenoBenchDataset(Dataset):
    """
    Dataset loader for PhenoBench.
    Expects:
    - Images folder containing RGB images (png/jpg).
    - Annotations folder containing 16-bit PNG semantic masks.
    """
    def __init__(self, image_folder_path, annotation_path, processor, label2id: dict):
        self.image_folder = image_folder_path
        self.annotation_path = annotation_path  # In PhenoBench, this is a folder, not a file
        self.processor = processor
        self.label2id = label2id

        # List all images
        self.image_files = glob.glob(os.path.join(self.image_folder, '*.png'))
        self.image_files.sort()

        # Filter pairs
        self.valid_files = []
        valid_count = 0
        for img_path in self.image_files:
            file_name = os.path.basename(img_path)
            # Mask assumption: same basename, definitely .png
            mask_name = os.path.splitext(file_name)[0] + '.png'
            mask_path = os.path.join(self.annotation_path, mask_name)

            if os.path.exists(mask_path):
                self.valid_files.append((img_path, mask_path))
                valid_count += 1
                if (config.MAX_IMAGES is not None) and (valid_count >= config.MAX_IMAGES):
                    break

        print(f'\tLoaded {len(self.valid_files)} valid image/mask pairs from "{self.image_folder}"')

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.valid_files[idx]
        file_name = os.path.basename(image_path)

        # Load Image
        image = Image.open(image_path).convert('RGB')

        # Load Semantic Mask (16-bit PNG)
        # Using cv2 to load as is (unchanged depth)
        semantic_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        width, height = image.size

        # Resize if too large
        if max(width, height) > config.MAX_INPUT_DIM:
            scale_factor = config.MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize image (Bilinear)
            image = image.resize(size=(new_width, new_height), resample=Image.BILINEAR)

            # Resize mask (Nearest Neighbor to preserve labels)
            semantic_mask = cv2.resize(
                src=semantic_mask,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )

            width, height = new_width, new_height

        target_size = (height, width)

        # Convert Semantic Mask to Instance Mask
        # We need to identify connected components for every class to create instances
        # Mask2Former expects an instance map where every object has a unique ID

        instance_map = np.full((height, width), 255, dtype=np.int32)
        instance_id_to_semantic_id = {}
        current_instance_id = 1

        # Get unique classes present in the mask (excluding 0/background)
        unique_classes = np.unique(semantic_mask)

        for cls_id in unique_classes:
            if cls_id == 0:
                continue  # Skip background

            # Create binary mask for this class
            class_binary_mask = (semantic_mask == cls_id).astype(np.uint8)

            # Find connected components to separate instances
            num_labels, labels_im = cv2.connectedComponents(class_binary_mask)

            # labels_im contains 0 for background, 1...N for components
            for label_idx in range(1, num_labels):
                if current_instance_id == 255:
                    current_instance_id += 1

                # Assign unique instance ID to this component
                instance_map[labels_im == label_idx] = current_instance_id

                # Map instance ID back to semantic class ID
                # Note: user config maps class strings to IDs, but PhenoBench masks
                # already contain 1,2,3,4. We assume the raw pixel values map
                # directly to the IDs defined in definitions.py
                instance_id_to_semantic_id[current_instance_id] = int(cls_id)

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
