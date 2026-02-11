import os

# Dataset Root
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/PhenoBench/'

# Raw Data Paths
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'train/images/')
TRAIN_ANNOTATIONS = os.path.join(DATASET_ROOT, 'train/semantics/')

VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'val/images/')
VAL_ANNOTATIONS = os.path.join(DATASET_ROOT, 'val/semantics/')

TEST_IMG_DIR = os.path.join(DATASET_ROOT, 'test/images/')
TEST_ANNOTATIONS = os.path.join(DATASET_ROOT, 'test/semantics/')

# Processed Data Output
PROCESSED_DIR = os.path.join(DATASET_ROOT, 'Processed/')

# select the annotation format
# this dataset has 2 type of annotations: yaml and png. The yaml annotations are better for distinguishing single
# instances, but the segmentation is very coarse, more lyke a polygon contour. The png annotations are more precise,
# but they do not distinguish between single instances, so they are more like a semantic segmentation.
# ANNOTATION_FORMAT = 'yaml'
ANNOTATION_FORMAT = 'png'

# Class Mapping
ID2LABEL = {
    0: 'background',
    1: 'crop',
    2: 'weed',
    3: 'partial-crop',
    4: 'partial-weed',
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
