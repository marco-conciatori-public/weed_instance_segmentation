# Global Training Configuration

MODEL_CHECKPOINT = 'facebook/mask2former-swin-large-coco-instance'
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
EPOCHS = 100
GRADIENT_ACCUMULATION = 2
MAX_INPUT_DIM = 1024
MAX_IMAGES = None  # Set to None for full dataset, or an integer for debugging
OUTPUT_DIR = 'output/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'models/'
