import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import config
from datasets.factory import get_dataset_and_config
from models.model_utils import load_model, plot_segmentation

MODEL_ID = 'mask2former_fine_tuned/2026-02-09_23-50-52/best_model/'
IMG_NAME = 'TestSorghumWeed (7).JPG'


def run_inference(image_path, model, processor, device):
    image = Image.open(image_path).convert('RGB')
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


def load_ground_truth(image_name: str,
                      target_size: tuple,
                      annotation_file: str,
                      img_dir: str,
                      label2id: dict,
                      ) -> dict | None:
    """
    Loads ground truth annotations and constructs a result dictionary
    compatible with the visualization function.
    """
    if not os.path.exists(annotation_file):
        print(f'Annotation file not found: {annotation_file}')
        return None

    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON: {e}')
        return None

    # Find entry for the specific image
    entry = next((item for item in data.values() if item['filename'] == image_name), None)

    if not entry:
        print(f'No annotation found for "{image_name}"')
        return None

    # Determine scale factor based on original image size vs target size
    # target_size is (Width, Height) from the PIL Image
    img_path = os.path.join(img_dir, image_name)
    if os.path.exists(img_path):
        with Image.open(img_path) as orig_img:
            orig_w, orig_h = orig_img.size
    else:
        # Fallback: assume the JSON coordinates match the target_size
        print('Warning: Original image file not found. Assuming 1:1 scale.')
        orig_w, orig_h = target_size

    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # Create empty mask (H, W)
    segmentation = np.zeros((target_h, target_w), dtype=np.int32)
    segments_info = []

    regions = entry.get('regions', [])
    current_instance_id = 1

    for region in regions:
        shape_attr = region['shape_attributes']
        region_attr = region['region_attributes']

        if shape_attr['name'] != 'polygon':
            continue

        class_name = region_attr.get('classname', None)
        if class_name not in label2id:
            continue

        label_id = label2id[class_name]

        # Get points and scale them
        all_x = shape_attr['all_points_x']
        all_y = shape_attr['all_points_y']

        scaled_points = []
        for x, y in zip(all_x, all_y):
            scaled_points.append([int(x * scale_x), int(y * scale_y)])

        points_np = np.array(scaled_points, dtype=np.int32)

        # Draw filled polygon on mask
        cv2.fillPoly(segmentation, [points_np], color=current_instance_id)

        # Add info
        segments_info.append({
            'id': current_instance_id,
            'label_id': label_id,
            'score': 1.0  # Ground truth is always 100% confident
        })

        current_instance_id += 1

    return {
        'segmentation': torch.from_numpy(segmentation),
        'segments_info': segments_info
    }


if __name__ == '__main__':
    model, proc, dev = load_model(MODEL_ID)
    _, ds_config = get_dataset_and_config(config.DATASET_LIST[0])
    img_path = os.path.join(ds_config.TEST_IMG_DIR, IMG_NAME)

    if os.path.exists(img_path):
        # 1. Run Inference
        img, res = run_inference(img_path, model, proc, dev)

        # 2. Load Ground Truth
        gt_res = load_ground_truth(
            image_name=IMG_NAME,
            target_size=img.size,
            annotation_file=ds_config.TEST_ANNOTATIONS,
            img_dir=ds_config.TEST_IMG_DIR,
            label2id=ds_config.LABEL2ID
        )

        # 3. Visualize
        if gt_res:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            # Plot Prediction
            plot_segmentation(axes[0], img, res, model, instance_mode=False, score_threshold=0.5)
            axes[0].set_title('Prediction')

            # Plot Ground Truth
            plot_segmentation(axes[1], img, gt_res, model, instance_mode=False)
            axes[1].set_title('Ground Truth')

            plt.tight_layout()
            plt.show()
        else:
            # Fallback to single plot if GT fails
            fig, ax = plt.subplots(figsize=(12, 12))
            plot_segmentation(ax, img, res, model, instance_mode=False, score_threshold=0.5)
            ax.set_title('Prediction')
            plt.show()

    else:
        print(f'Image not found at {img_path}')
