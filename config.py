# Global Configuration

# Training
MODEL_CHECKPOINT = 'facebook/mask2former-swin-large-coco-instance'
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
EPOCHS = 100
GRADIENT_ACCUMULATION = 2

# Data
MAX_INPUT_DIM = 1024
MAX_IMAGES = None  # Set to None for full dataset, or an integer for debugging
DATASET_LIST = [
    'sorghum_weed',
    'pheno_bench',
    'crop_weed',
]
FORCE_PREPROCESSING = True

# Output Directories
OUTPUT_DIR = 'C:/Users/Marco/GIT/weed_instance_segmentation/output/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'models/'
