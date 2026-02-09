import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import config
from datasets.factory import get_dataset_config


def visualize_dataset(image_folder: str, annotation_file: str) -> None:
    if not os.path.exists(annotation_file):
        print(f'Error: Annotation file not found at {annotation_file}')
        return

    print('Loading annotations...')
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    class_colors = {'Sorghum': 'lime', 'BLweed': 'red', 'Grass': 'blue', 'default': 'yellow'}
    entries = list(data.values())
    valid_img_counter = 0

    print(f'Searching for valid images in {image_folder}...')
    for entry in entries:
        file_name = entry['filename']
        image_path = os.path.join(image_folder, file_name)

        if not os.path.exists(image_path):
            continue

        print(f'Displaying: {file_name}')
        try:
            image = Image.open(image_path)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)

            regions = entry.get('regions', [])
            legend_patches = {}

            for region in regions:
                shape_attr = region['shape_attributes']
                region_attr = region['region_attributes']

                if shape_attr['name'] == 'polygon':
                    all_x = shape_attr['all_points_x']
                    all_y = shape_attr['all_points_y']
                    poly_points = list(zip(all_x, all_y))

                    class_name = region_attr.get('classname', 'default')
                    color = class_colors.get(class_name, class_colors['default'])

                    poly = Polygon(poly_points, closed=True, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
                    ax.add_patch(poly)

                    if class_name not in legend_patches:
                        legend_patches[class_name] = poly

            ax.set_title(f'Annotation: {file_name}')
            ax.axis('off')
            if legend_patches:
                ax.legend(list(legend_patches.values()), list(legend_patches.keys()), loc='upper right')

            plt.tight_layout()
            plt.show()
            valid_img_counter += 1

        except Exception as e:
            print(f'Error processing image: {e}')

    if valid_img_counter == 0:
        print('No valid images found.')


if __name__ == '__main__':
    ds_config = get_dataset_config(config.DATASET_LIST[0])
    visualize_dataset(ds_config.TEST_IMG_DIR, ds_config.TEST_JSON)
