from datasets.crop_weed.definitions import ANNOTATION_FORMAT, IMG_DIR, ANNOTATIONS

# Dynamic import based on configuration
if ANNOTATION_FORMAT == 'png':
    from datasets.crop_weed.annotation_dependent_implementations.visualize_png_annotations import visualize_dataset
elif ANNOTATION_FORMAT == 'yaml':
    from datasets.crop_weed.annotation_dependent_implementations.visualize_yaml_annotations import visualize_dataset
else:
    raise ValueError(f'Unknown ANNOTATION_FORMAT "{ANNOTATION_FORMAT}" in "datasets/crop_weed/definitions.py". '
                     f'Supported formats are "png" and "yaml".')

if __name__ == '__main__':
    visualize_dataset(IMG_DIR, ANNOTATIONS)
