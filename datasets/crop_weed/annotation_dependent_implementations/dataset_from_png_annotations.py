import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class CropWeedDataset(Dataset):
    """
    Dataset loader for Crop Weed Field Image Dataset (CWFID).
    Expects:
    - Images folder containing RGB images (png/jpg).
    - Annotations folder containing RGB PNG semantic masks (Red=Weed, Green=Crop).
    """
    def __init__(self, image_folder_path, annotation_path, processor, label2id: dict):
        self.image_folder = image_folder_path
        self.annotation_path = annotation_path
        self.processor = processor
        self.label2id = label2id

        # List all images
        self.image_files = glob.glob(os.path.join(self.image_folder, '*.png'))
        self.image_files.sort()

        # Filter valid pairs
        self.valid_files = []
        valid_count = 0
        for img_path in self.image_files:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            image_number = base_name.split('_')[0]
            mask_name = image_number + '_annotation.png'
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

        # Load Mask (Use OpenCV to preserve values)
        # CWFID masks are typically RGB: Red=Weed, Green=Crop, Black=Background
        mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        width, height = image.size

        # Resize if too large
        if max(width, height) > config.MAX_INPUT_DIM:
            scale_factor = config.MAX_INPUT_DIM / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            # Resize mask using Nearest Neighbor to preserve color codes
            mask_rgb = cv2.resize(mask_rgb, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            width, height = new_width, new_height

        target_size = (height, width)

        # --- Create Instance Map ---
        # Initialize with 255 (ignore/background)
        instance_map = np.full((height, width), 255, dtype=np.int32)
        instance_id_to_semantic_id = {}
        current_instance_id = 1

        # Define color thresholds/values
        # Crop: Green (0, 255, 0) -> Class ID 0 (from definitions.py)
        # Weed: Red (255, 0, 0)   -> Class ID 1 (from definitions.py)

        # We process classes one by one
        # Map: Color -> Class Name -> Class ID
        color_map = {
            'crop': {'color': [0, 255, 0], 'id': self.label2id.get('crop', 0)},
            'weed': {'color': [255, 0, 0], 'id': self.label2id.get('weed', 1)}
        }

        for cls_name, cls_info in color_map.items():
            color = np.array(cls_info['color'])
            semantic_id = cls_info['id']

            # Create binary mask for this class
            # exact match
            class_mask = np.all(mask_rgb == color, axis=-1).astype(np.uint8)

            # Find connected components to create distinct instances from the semantic mask
            num_labels, labels_im = cv2.connectedComponents(class_mask)

            # labels_im: 0 is background (within this class mask), 1...N are components
            for label_idx in range(1, num_labels):
                if current_instance_id == 255:
                    current_instance_id += 1

                # Assign unique instance ID
                instance_map[labels_im == label_idx] = current_instance_id
                instance_id_to_semantic_id[current_instance_id] = semantic_id
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
