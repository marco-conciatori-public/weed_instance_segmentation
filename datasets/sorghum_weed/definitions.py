import os

# Dataset Root
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'

# Raw Data Paths
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'Train/')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'Annotations/TrainSorghumWeed_json.json')

VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'Validate/')
VAL_JSON = os.path.join(DATASET_ROOT, 'Annotations/ValidateSorghumWeed_json.json')

TEST_IMG_DIR = os.path.join(DATASET_ROOT, 'Test/')
TEST_JSON = os.path.join(DATASET_ROOT, 'Annotations/TestSorghumWeed_json.json')

# Processed Data Output
PROCESSED_DIR = os.path.join(DATASET_ROOT, 'Processed/')

# Class Mapping
ID2LABEL = {
    0: "Sorghum",
    1: "BLweed",
    2: "Grass"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
