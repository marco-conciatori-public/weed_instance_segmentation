import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config
from datasets.pheno_bench import definitions as pb_definitions


def visualize_dataset(image_folder: str, annotation_folder: str) -> None:
    if not os.path.exists(annotation_folder):
        print(f'Error: Annotation folder not found at {annotation_folder}')
        return

    # Define colors for classes (matching definitions.py logic)
    # 0: background, 1: crop, 2: weed, 3: partial-crop, 4: partial-weed
    label_colors = {
        0: [0, 0, 0],        # Background (Black)
        1: [0, 255, 0],      # Crop (Green)
        2: [255, 0, 0],      # Weed (Red)
        3: [0, 255, 255],  # Partial-Crop (Yellow)
        4: [255, 0, 255]   # Partial-Weed (Purple)
    }

    label_names = {
        0: 'background',
        1: 'crop',
        2: 'weed',
        3: 'partial-crop',
        4: 'partial-weed'
    }

    print(f'Searching for images in {image_folder}...')
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    image_files.sort()

    valid_img_counter = 0

    for img_path in image_files:
        file_name = os.path.basename(img_path)
        mask_name = os.path.splitext(file_name)[0] + '.png'
        mask_path = os.path.join(annotation_folder, mask_name)

        if not os.path.exists(mask_path):
            continue

        print(f'Displaying: {file_name}')
        try:
            # Load Image (RGB)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # Load Mask (1-channel)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # Create color overlay
            h, w = mask.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)

            unique_labels = np.unique(mask)
            legend_handles = []

            for label_id in unique_labels:
                if label_id == 0:
                    continue

                color = label_colors.get(label_id, [255, 255, 0])
                color_mask[mask == label_id] = color

                # Add to legend
                patch = Patch(color=np.array(color)/255.0, label=label_names.get(label_id, f'Class {label_id}'))
                legend_handles.append(patch)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.imshow(color_mask, alpha=0.5)  # Overlay with transparency

            ax.set_title(f'Annotation: {file_name}')
            ax.axis('off')

            if legend_handles:
                ax.legend(handles=legend_handles, loc='upper right')

            plt.tight_layout()
            plt.show()
            valid_img_counter += 1

        except Exception as e:
            print(f'Error processing image {file_name}: {e}')

        # Limit to first config.MAX_IMAGES examples for visualization check
        if config.MAX_IMAGES is not None and valid_img_counter >= config.MAX_IMAGES:
            break

    if valid_img_counter == 0:
        print('No valid image/mask pairs found.')


if __name__ == '__main__':
    visualize_dataset(pb_definitions.TRAIN_IMG_DIR, pb_definitions.TRAIN_JSON)
