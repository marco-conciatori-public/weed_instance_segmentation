from datasets.crop_weed.definitions import ANNOTATION_FORMAT

# Dynamic import based on configuration
if ANNOTATION_FORMAT == 'png':
    from datasets.crop_weed.annotation_dependent_implementations.dataset_from_png_annotations import CropWeedDataset
elif ANNOTATION_FORMAT == 'yaml':
    from datasets.crop_weed.annotation_dependent_implementations.dataset_from_yaml_annotations import CropWeedDataset
else:
    raise ValueError(f'Unknown ANNOTATION_FORMAT "{ANNOTATION_FORMAT}" in "datasets/crop_weed/definitions.py". '
                     f'Supported formats are "png" and "yaml".')
