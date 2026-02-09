import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from datasets.sorghum_weed import definitions as ds_config

MODEL_ID = 'mask2former_fine_tuned/2026-02-09_19-50-56/best_model/'
IMG_NAME = 'TestSorghumWeed (7).JPG'


def load_model(model_id):
    model_path = config.MODELS_OUTPUT_DIR + model_id
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

    # resize masks back to the image size (which might be resized)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return image, result


def plot_segmentation(ax, image, result: dict, model, instance_mode: bool = True, score_threshold: float = 0.0) -> None:
    segmentation = result['segmentation']
    if hasattr(segmentation, 'cpu'):
        segmentation = segmentation.cpu().numpy()

    segments = result['segments_info']

    # Filter segments by score
    valid_segments = [s for s in segments if s.get('score', 1.0) >= score_threshold]
    num_instances = len(valid_segments)

    ax.imshow(image)

    # Create an empty colored mask (RGBA for transparency)
    color_mask = np.zeros((*segmentation.shape, 4))
    contours_to_draw = []
    legend_patches = []

    # Track counts for each class label to assign indices
    class_counts = {}
    seen_classes_in_legend = set()

    # Color Setup
    if instance_mode:
        num_colors_needed = num_instances
    else:
        unique_label_ids = sorted(list(set(s['label_id'] for s in valid_segments)))
        num_colors_needed = len(unique_label_ids)
        class_color_index_map = {lbl: idx for idx, lbl in enumerate(unique_label_ids)}

    # Select Colormap
    if num_colors_needed <= 20:
        cmap = plt.get_cmap('tab20')
        color_palette = [cmap(i) for i in range(max(num_colors_needed, 1))]
    else:
        cmap = plt.get_cmap('nipy_spectral')
        color_palette = [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]

    for i, segment in enumerate(valid_segments):
        segment_id = segment['id']
        label_id = segment['label_id']

        # Get Label Text
        if hasattr(model.config, 'id2label') and label_id in model.config.id2label:
            label_text = model.config.id2label[label_id]
        else:
            label_text = ds_config.ID2LABEL.get(label_id, f"Class {label_id}")

        count = class_counts.get(label_text, 0) + 1
        class_counts[label_text] = count

        # Determine Color and Legend Label
        should_add_to_legend = False
        display_label = label_text

        if instance_mode:
            rgb = color_palette[i % len(color_palette)][:3]
            display_label = f"{label_text} {count}"
            should_add_to_legend = True
        else:
            color_idx = class_color_index_map[label_id]
            rgb = color_palette[color_idx % len(color_palette)][:3]
            if label_id not in seen_classes_in_legend:
                should_add_to_legend = True
                seen_classes_in_legend.add(label_id)

        # 1. Prepare Fill
        mask_bool = (segmentation == segment_id)
        fill_color = [*rgb, 0.4]
        color_mask[mask_bool] = fill_color

        # 2. Prepare Contour
        contour_color = [*rgb, 1.0]
        contours_to_draw.append((mask_bool, contour_color))

        if should_add_to_legend:
            patch = mpatches.Patch(color=fill_color, label=display_label)
            legend_patches.append(patch)

    # Show the transparent fills
    ax.imshow(color_mask)

    # Draw contours on top
    for mask, color in contours_to_draw:
        try:
            # Check if mask is not empty before contouring
            if mask.any():
                ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)
        except Exception:
            pass

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', framealpha=0.8)
    ax.axis('off')


if __name__ == '__main__':
    model, proc, dev = load_model(MODEL_ID)
    img_path = os.path.join(ds_config.TEST_IMG_DIR, IMG_NAME)

    if os.path.exists(img_path):
        img, res = run_inference(img_path, model, proc, dev)
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_segmentation(ax, img, res, model, score_threshold=0.5)
        plt.show()
    else:
        print(f"Image not found at {img_path}")
