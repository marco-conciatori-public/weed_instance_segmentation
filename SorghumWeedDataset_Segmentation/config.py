import os

# Paths
DATASET_ROOT = 'F:/LAVORO/Miningful/weed_segmentation_dataset/SorghumWeedDataset_Segmentation/'
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'Train/')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'Annotations/TrainSorghumWeed_json.json')
VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'Validate/')
VAL_JSON = os.path.join(DATASET_ROOT, 'Annotations/ValidateSorghumWeed_json.json')
TEST_IMG_DIR = os.path.join(DATASET_ROOT, 'Test/')
TEST_JSON = os.path.join(DATASET_ROOT, 'Annotations/TestSorghumWeed_json.json')
PROCESSED_DIR = os.path.join(DATASET_ROOT, 'Processed/')

OUTPUT_DIR = 'models/mask2former_fine_tuned'

# Model Config
MODEL_CHECKPOINT = 'facebook/mask2former-swin-large-coco-instance'
BATCH_SIZE = 2  # Reduce to 1 if OOM
LEARNING_RATE = 5e-5
EPOCHS = 10
GRADIENT_ACCUMULATION = 2
MAX_INPUT_DIM = 1024  # Resize images larger than this to save VRAM
MAX_IMAGES = 10  # Set to None for full dataset, or a number for debugging

# Class Mapping (Internal ID -> Name)
# Mask2Former uses background implicitly
ID2LABEL = {
    0: "Sorghum",
    1: "BLweed",
    2: "Grass"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}