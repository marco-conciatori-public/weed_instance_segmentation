import os

# Dataset Root
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/Crop Weed Field Image Dataset/'

# Raw Data Paths
IMG_DIR = os.path.join(DATASET_ROOT, 'images/')
ANNOTATIONS = os.path.join(DATASET_ROOT, 'annotations/')

# Processed Data Output
PROCESSED_DIR = os.path.join(DATASET_ROOT, 'Processed/')

# This parameter controls how the dataset is split in train/val/test. If the validation or test split is 0, then that
# set should not be created, and the data should be split only in the remaining sets. For example, if
# TRAIN_VAL_TEST_SPLIT = [0.8, 0, 0.2], the dataset should be split only in train and test sets, with 80% of the data
# in train and 20% in test, and no validation set should be created.
TRAIN_VAL_TEST_SPLIT = [0.8, 0.2, 0]
# check if the split sums to 1.0 +/- a small epsilon for floating point precision
if abs(sum(TRAIN_VAL_TEST_SPLIT) - 1.0) > 1e-6:
    raise ValueError(f'TRAIN_VAL_TEST_SPLIT must sum to 1.0, but got {sum(TRAIN_VAL_TEST_SPLIT)}')

# select the annotation format
# this dataset has 2 type of annotations: YAML and png. The YAML annotations are better for distinguishing single
# instances, but the segmentation is very coarse, more like a polygon contour. The png annotations are more precise,
# but they do not distinguish between single instances, so they are more like a semantic segmentation.
# ANNOTATION_FORMAT = 'yaml'
ANNOTATION_FORMAT = 'png'

# Class Mapping
ID2LABEL = {
    0: 'crop',
    1: 'weed',
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
