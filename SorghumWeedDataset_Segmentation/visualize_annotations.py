import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

DATASET_FOLDER_PATH = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'
ANNOTATION_FOLDER_PATH = DATASET_FOLDER_PATH + 'Annotations/'
TRAIN_FOLDER_PATH = DATASET_FOLDER_PATH + 'Train/'
VALIDATION_FOLDER_PATH = DATASET_FOLDER_PATH + 'Validation/'
TEST_FOLDER_PATH = DATASET_FOLDER_PATH + 'Test/'
TRAIN_ANNOTATIONS_FILE = ANNOTATION_FOLDER_PATH + 'TrainSorghumWeed_json.json'
VALIDATION_ANNOTATIONS_FILE = ANNOTATION_FOLDER_PATH + 'ValidationSorghumWeed_json.json'
TEST_ANNOTATIONS_FILE = ANNOTATION_FOLDER_PATH + 'TestSorghumWeed_json.json'


def visualize_dataset(image_folder: str, annotation_file: str) -> None:
    """
    Visualizes one image from the dataset with VIA (VGG Image Annotator) polygon annotations.
    """

    # 1. Check if paths exist
    if not os.path.exists(annotation_file):
        print(f'Error: Annotation file not found at {annotation_file}')
        return

    if not os.path.exists(image_folder):
        print(f'Error: Image folder not found at {image_folder}')
        return

    # 2. Load the JSON data
    print('Loading annotations...')
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # 3. Define color map for classes
    class_colors = {
        'Sorghum': 'lime',  # Green for crop
        'BLweed': 'red',  # Red for Broadleaf weeds
        'Grass': 'blue',  # Blue for Grass
        'default': 'yellow'  # Fallback
    }

    # 4. Find an image that actually exists in the folder
    # Get all dictionary values (file entries)
    entries = list(data.values())
    valid_img_counter = 0

    print(f'Searching for a valid image in "{image_folder}"...')
    for entry in entries:
        file_name = entry['filename']
        print(f'\tChecking for file: {file_name}')
        image_path = os.path.join(image_folder, file_name)

        if not os.path.exists(image_path):
            continue

        print(f'\t\tDisplaying: {entry['filename']}')

        # 5. Load and Plot Image
        try:
            image = Image.open(image_path)

            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image)

            # 6. Draw Polygons
            regions = entry.get('regions', [])

            legend_patches = {}  # To store handles for the legend

            for region in regions:
                # Get shape attributes
                shape_attr = region['shape_attributes']
                region_attr = region['region_attributes']

                if shape_attr['name'] == 'polygon':
                    all_x = shape_attr['all_points_x']
                    all_y = shape_attr['all_points_y']

                    # Zip x and y coordinates into (x, y) tuples
                    poly_points = list(zip(all_x, all_y))

                    # Determine class and color
                    class_name = region_attr.get('classname', 'default')
                    color = class_colors.get(class_name, class_colors['default'])

                    # Create the polygon patch
                    # alpha controls transparency (0.3 is 30% opaque)
                    poly = Polygon(
                        poly_points,
                        closed=True,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )

                    ax.add_patch(poly)

                    # Add border only for clearer visibility
                    border = Polygon(
                        poly_points,
                        closed=True,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        alpha=0.9,
                    )
                    ax.add_patch(border)

                    # Store for legend
                    if class_name not in legend_patches:
                        legend_patches[class_name] = poly

            # 7. Final Formatting
            ax.set_title(f'Annotation: {entry['filename']}', fontsize=14)
            ax.axis('off')  # Hide axes numbers

            # Create a custom legend
            handles = list(legend_patches.values())
            labels = list(legend_patches.keys())
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=12, framealpha=0.8)

            plt.tight_layout()
            plt.show()
            valid_img_counter += 1

        except Exception as e:
            print(f'\t\tAn error occurred while processing the image: {e}')

    if valid_img_counter == 0:
        print('Could not find any images from the JSON inside the folder.')
        return


if __name__ == '__main__':
    visualize_dataset(TEST_FOLDER_PATH, TEST_ANNOTATIONS_FILE)
