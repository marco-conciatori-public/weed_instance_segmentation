import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

import config
from datasets.factory import get_dataset_config
from models.model_utils import load_model, plot_segmentation

MODEL_ID = 'mask2former_fine_tuned/2026-02-09_19-50-56/best_model/'
IMG_NAME = 'TestSorghumWeed (7).JPG'


def run_inference(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) > config.MAX_INPUT_DIM:
        scale = config.MAX_INPUT_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # resize masks back to the image size (which might be resized)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return image, result


if __name__ == '__main__':
    model, proc, dev = load_model(MODEL_ID)
    ds_config = get_dataset_config(config.DATASET_LIST[0])
    img_path = os.path.join(ds_config.TEST_IMG_DIR, IMG_NAME)

    if os.path.exists(img_path):
        img, res = run_inference(img_path, model, proc, dev)
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_segmentation(ax, img, res, model, instance_mode=False, score_threshold=0.5)
        plt.show()
    else:
        print(f"Image not found at {img_path}")
