import os
import glob
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

import config
from datasets.crop_weed import definitions as cw_definitions


def visualize_dataset(image_folder: str, annotation_folder: str) -> None:
    if not os.path.exists(annotation_folder):
        print(f'Error: Annotation folder not found at {annotation_folder}')
        return

    # Define Display Colors
    display_colors = {
        'crop': 'lime',
        'weed': 'red',
        'default': 'yellow'
    }

    # List all YAML files
    yaml_files = glob.glob(os.path.join(annotation_folder, '*.yaml'))
    yaml_files.sort()

    print(f'Found {len(yaml_files)} annotation files in {annotation_folder}...')

    valid_img_counter = 0

    for yaml_path in yaml_files:
        try:
            # Load YAML
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                continue

            # Get image filename from annotation
            img_filename = data.get('filename')
            if not img_filename:
                continue

            img_path = os.path.join(image_folder, img_filename)

            if not os.path.exists(img_path):
                print(f"Image not found for annotation {os.path.basename(yaml_path)}: {img_filename}")
                continue

            print(f'Displaying: {img_filename}')

            # Load Image for display
            from PIL import Image
            image = Image.open(img_path)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)

            regions = data.get('annotation', [])
            legend_patches = {}

            for region in regions:
                type_name = region.get('type', 'default')

                points_dict = region.get('points', {})
                xs = points_dict.get('x', [])
                ys = points_dict.get('y', [])

                if len(xs) != len(ys) or len(xs) < 3:
                    continue

                poly_points = list(zip(xs, ys))

                color = display_colors.get(type_name, display_colors['default'])

                # Create Polygon Patch
                poly = Polygon(
                    poly_points,
                    closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.4
                )
                ax.add_patch(poly)

                if type_name not in legend_patches:
                    legend_patches[type_name] = poly

            ax.set_title(f'Annotation: {os.path.basename(yaml_path)}\nImage: {img_filename}')
            ax.axis('off')

            if legend_patches:
                # Create custom legend handles to ensure colors match labels neatly
                handles = []
                labels = []
                for name, patch in legend_patches.items():
                    # We create a simple proxy artist for the legend
                    handles.append(Patch(color=patch.get_facecolor(), label=name))
                    labels.append(name)
                ax.legend(handles, labels, loc='upper right')

            plt.tight_layout()
            plt.show()
            valid_img_counter += 1

        except Exception as e:
            print(f'Error processing annotation {os.path.basename(yaml_path)}: {e}')

        if config.MAX_IMAGES is not None and valid_img_counter >= config.MAX_IMAGES:
            break

    if valid_img_counter == 0:
        print('No valid image/yaml pairs found.')


if __name__ == '__main__':
    visualize_dataset(cw_definitions.IMG_DIR, cw_definitions.ANNOTATIONS)
