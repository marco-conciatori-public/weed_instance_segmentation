import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.sorghum_weed import definitions as ds_config


def load_model(model_path):
    if not os.path.exists(model_path):
        model_path = config.MODEL_CHECKPOINT
    print(f'Loading {model_path}...')
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, processor, device


def run_inference(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) > config.MAX_INPUT_DIM:
        scale = config.MAX_INPUT_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return image, result


def plot_segmentation(ax, image, result, model, instance_mode=True) -> None:
    segmentation = result['segmentation'].cpu().numpy()
    segments = result['segments_info']
    ax.imshow(image)

    color_mask = np.zeros((*segmentation.shape, 4))
    legend_patches = []

    # Generate colors
    unique_ids = sorted(list(set(s['label_id'] for s in segments)))
    cmap = plt.get_cmap('tab20')

    for i, seg in enumerate(segments):
        label_id = seg['label_id']
        label_text = model.config.id2label.get(label_id, str(label_id))

        # Color logic
        color_idx = i if instance_mode else unique_ids.index(label_id)
        rgb = cmap(color_idx % 20)[:3]

        mask = (segmentation == seg['id'])
        color_mask[mask] = [*rgb, 0.4]  # Fill

        patch = mpatches.Patch(color=[*rgb, 0.4], label=label_text)
        if label_text not in [p.get_label() for p in legend_patches]:
            legend_patches.append(patch)

    ax.imshow(color_mask)
    ax.legend(handles=legend_patches, loc='upper right')
    ax.axis('off')


if __name__ == '__main__':
    # Example Usage
    MODEL_PATH = 'models/mask2former_fine_tuned/YOUR_RUN_TIMESTAMP/best_model'
    IMG_NAME = 'TestSorghumWeed (7).JPG'

    model, proc, dev = load_model(MODEL_PATH)
    img_path = os.path.join(ds_config.TEST_IMG_DIR, IMG_NAME)

    if os.path.exists(img_path):
        img, res = run_inference(img_path, model, proc, dev)
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_segmentation(ax, img, res, model)
        plt.show()
