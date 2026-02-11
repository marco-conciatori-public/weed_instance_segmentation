import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config
from datasets.crop_weed import definitions as cw_definitions


def visualize_dataset(image_folder: str, annotation_folder: str) -> None:
    if not os.path.exists(annotation_folder):
        print(f'Error: Annotation folder not found at {annotation_folder}')
        return

    # Define CWFID specific colors
    # Annotations are: Red=Weed, Green=Crop
    # We map them to display colors
    display_colors = {
        'crop': [0, 255, 0],   # Green
        'weed': [255, 0, 0],   # Red
        'background': [0, 0, 0]
    }

    print(f'Searching for images in {image_folder}...')
    image_files = glob.glob(os.path.join(image_folder, '*.png'))
    image_files.sort()

    valid_img_counter = 0
    for img_path in image_files:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        image_number = base_name.split('_')[0]
        mask_name = image_number + '_annotation.png'
        mask_path = os.path.join(annotation_folder, mask_name)

        if not os.path.exists(mask_path):
            continue

        print(f'Displaying: {file_name}')
        try:
            # Load Image (RGB)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # Load Mask (RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            # Create visualization overlay
            h, w, _ = image.shape
            overlay = np.zeros_like(image)

            # Masks for classes
            # CWFID standard: Crop is (0, 255, 0), Weed is (255, 0, 0)
            crop_mask = np.all(mask == [0, 255, 0], axis=-1)
            weed_mask = np.all(mask == [255, 0, 0], axis=-1)

            # Apply colors to overlay
            overlay[crop_mask] = display_colors['crop']
            overlay[weed_mask] = display_colors['weed']

            # Blend
            alpha = 0.5
            # Create a combined mask of any annotation
            has_annotation = crop_mask | weed_mask

            # Copy image to output
            output = image.copy()
            # Blend only where we have annotations
            output[has_annotation] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[has_annotation]

            # Legend
            legend_handles = [
                Patch(color=np.array(display_colors['crop'])/255.0, label='Crop'),
                Patch(color=np.array(display_colors['weed'])/255.0, label='Weed')
            ]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(output)
            ax.set_title(f'Annotation: {file_name}')
            ax.axis('off')
            ax.legend(handles=legend_handles, loc='upper right')

            plt.tight_layout()
            plt.show()
            valid_img_counter += 1

        except Exception as e:
            print(f'Error processing image {file_name}: {e}')

        if config.MAX_IMAGES is not None and valid_img_counter >= config.MAX_IMAGES:
            break

    if valid_img_counter == 0:
        print('No valid image/mask pairs found.')


if __name__ == '__main__':
    visualize_dataset(cw_definitions.IMG_DIR, cw_definitions.ANNOTATIONS)