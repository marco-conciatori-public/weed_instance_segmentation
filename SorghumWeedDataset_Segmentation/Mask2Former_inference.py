import os
import cv2
import json
import torch
import config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

SHOW_GROUND_TRUTH = True
INSTANCE_MODE = False
# IMAGE_NAME = 'TestSorghumWeed (7).JPG'
IMAGE_NAME = 'TestSorghumWeed (14).JPG'


def load_model():
    """
    Loads the Mask2Former model and processor.
    """

    # Checkpoint paths
    model_id_or_path = 'models/mask2former_fine_tuned/2026-02-06_02-49-29/best_model'

    # Fallback to config if local path doesn't exist
    if not os.path.exists(model_id_or_path):
        print(f"Path '{model_id_or_path}' not found. Falling back to config default.")
        model_id_or_path = config.MODEL_CHECKPOINT

    print(f'Loading {model_id_or_path}...')

    processor = AutoImageProcessor.from_pretrained(model_id_or_path, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id_or_path)

    # Use GPU if available
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_name)
    print(f'Model loaded on {device_name.upper()}')

    return model, processor, device_name


def run_inference(image_path: str, model, processor, device_name: str):
    """
    Runs inference on a single image.
    """
    # Load Image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > config.MAX_INPUT_DIM:
        scale_factor = config.MAX_INPUT_DIM / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    print(f'Processing image size: {image.size}')

    # Preprocess
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device_name) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    # resize masks back to original image (target_sizes height and width)
    result = processor.post_process_instance_segmentation(
        outputs=outputs,
        target_sizes=[image.size[::-1]]  # (Height, Width)
    )[0]

    return image, result


def load_ground_truth(image_name: str, target_size: tuple) -> dict | None:
    """
    Loads ground truth annotations from the TEST_JSON file.
    Constructs a result dictionary compatible with the visualization function.

    Args:
        image_name: The filename of the image (e.g., '123.jpg').
        target_size: The (width, height) of the image being displayed/processed.
                     Used to scale polygon coordinates.
    """
    json_path = config.TEST_JSON

    if not os.path.exists(json_path):
        print(f"Ground truth JSON not found at {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

    # Find entry for the specific image
    entry = next((item for item in data.values() if item['filename'] == image_name), None)

    if not entry:
        print(f"No annotation found for '{image_name}' in {json_path}")
        return None

    # Determine scale factor based on original image size vs target size
    img_path = os.path.join(config.TEST_IMG_DIR, image_name)
    if os.path.exists(img_path):
        with Image.open(img_path) as orig_img:
            orig_w, orig_h = orig_img.size
    else:
        # Fallback: assume the JSON coordinates match the target_size if we can't verify
        print("Warning: Could not find original image file to verify scale. Assuming 1:1 scale with target.")
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
        if class_name not in config.LABEL2ID:
            continue

        label_id = config.LABEL2ID[class_name]

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


def plot_segmentation(ax,
                      image,
                      result: dict,
                      model,
                      confidence_threshold: float,
                      instance_mode: bool,
                      title: str,
                      ) -> None:
    """
    Helper function to plot segmentation on a specific matplotlib axis.
    """
    segmentation = result['segmentation']
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    segments_info = result['segments_info']

    ax.imshow(image)

    # Create an empty colored mask (RGBA for transparency)
    color_mask = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    legend_patches = []
    contours_to_draw = []

    # Track counts for each class label to assign indices
    class_counts = {}
    seen_classes_in_legend = set()

    # Filter segments
    valid_segments = [s for s in segments_info if s.get('score', 0) >= confidence_threshold]
    num_instances = len(valid_segments)

    print(f"[{title}] Displaying {num_instances} segments.")

    # Color Setup
    if instance_mode:
        num_colors_needed = num_instances
    else:
        unique_label_ids = sorted(list(set(s['label_id'] for s in valid_segments)))
        num_colors_needed = len(unique_label_ids)
        # Create a map: label_id -> index in the color list
        class_color_index_map = {lbl: idx for idx, lbl in enumerate(unique_label_ids)}

    # Select Colormap based on number of colors needed
    if num_colors_needed <= 20:
        cmap = plt.get_cmap('tab20')
        color_palette = [cmap(i) for i in range(max(num_colors_needed, 1))]
    else:
        cmap = plt.get_cmap('nipy_spectral')
        color_palette = [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]

    for i, segment in enumerate(valid_segments):
        segment_id = segment['id']
        label_id = segment['label_id']

        # Safe label lookup
        if hasattr(model.config, 'id2label') and label_id in model.config.id2label:
            label_text = model.config.id2label[label_id]
        else:
            label_text = config.ID2LABEL.get(label_id, f"Class {label_id}")

        count = class_counts.get(label_text, 0) + 1
        class_counts[label_text] = count

        # Determine Color and Label
        if instance_mode:
            rgb = color_palette[i][:3]
            display_label = f"{label_text} {count}"
            should_add_to_legend = True
        else:
            color_idx = class_color_index_map[label_id]
            rgb = color_palette[color_idx][:3]
            display_label = label_text

            # Only add to legend if we haven't seen this class yet
            if label_id not in seen_classes_in_legend:
                should_add_to_legend = True
                seen_classes_in_legend.add(label_id)
            else:
                should_add_to_legend = False

        # Drawing Logic
        # 1. Fill
        fill_alpha = 0.4
        fill_color = np.concatenate([rgb, [fill_alpha]])

        # Apply fill color to the mask location
        mask_bool = (segmentation == segment_id)
        color_mask[mask_bool] = fill_color

        # 2. Contour
        contour_alpha = 1.0
        contour_color = np.concatenate([rgb, [contour_alpha]])
        contours_to_draw.append((mask_bool, contour_color))

        if should_add_to_legend:
            patch = mpatches.Patch(color=fill_color, label=display_label)
            legend_patches.append(patch)

    # Show the transparent fills
    ax.imshow(color_mask)

    # Draw contours on top of the fills
    for mask, color in contours_to_draw:
        try:
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)
        except Exception:
            # Handles cases where mask might be empty or singular
            pass

    ax.axis('off')
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', framealpha=0.8)

    ax.set_title(title, fontsize=16)


def visualize_result(image,
                     result: dict,
                     model,
                     ground_truth_result: dict = None,
                     confidence_threshold: float = 0.5,
                     instance_mode: bool = True,
                     ) -> None:
    """
    Visualizes the instance segmentation masks.
    If ground_truth_result is provided, plots side-by-side.
    """

    if ground_truth_result:
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        plot_segmentation(axes[0], image, result, model, confidence_threshold, instance_mode, "Model Prediction")
        plot_segmentation(
            axes[1],
            image,
            ground_truth_result,
            model,
            confidence_threshold,
            instance_mode,
            "Ground Truth",
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_segmentation(
            ax,
            image,
            result,
            model,
            confidence_threshold,
            instance_mode,
            f"Prediction ({'Instance' if instance_mode else 'Class'})",
        )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load Model
    model, processor, device_name = load_model()

    # Image Setup
    DATA_FOLDER_PATH = config.TEST_IMG_DIR
    image_full_path = os.path.join(DATA_FOLDER_PATH, IMAGE_NAME)

    if not os.path.exists(image_full_path):
        print(f"Image not found at {image_full_path}. Please check path.")
    else:
        image, result = run_inference(
            image_path=image_full_path,
            model=model,
            processor=processor,
            device_name=device_name,
        )

        # Handle Ground Truth
        gt_result = None
        if SHOW_GROUND_TRUTH:
            print(f"Attempting to load ground truth for {IMAGE_NAME}...")
            gt_result = load_ground_truth(IMAGE_NAME, image.size)

        # Show Output
        visualize_result(
            image=image,
            result=result,
            model=model,
            ground_truth_result=gt_result,
            instance_mode=INSTANCE_MODE
        )
