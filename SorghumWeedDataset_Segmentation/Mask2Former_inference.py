import torch
import config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor


def load_model():
    """
    Loads the Mask2Former model and processor
    with the Swin-Large backbone for maximum accuracy
    """

    # 'facebook/mask2former-swin-large-coco-instance' is the high-accuracy
    # checkpoint fine-tuned for instance segmentation.
    # model_id_or_path = 'facebook/mask2former-swin-large-coco-instance'
    model_id_or_path = 'models/mask2former_fine_tuned/2026-02-06_02-49-29/best_model'
    # model_id_or_path = 'models/mask2former_fine_tuned/2026-02-04_20-56-07/final_model'
    print(f'Loading {model_id_or_path}...')

    processor = AutoImageProcessor.from_pretrained(model_id_or_path, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id_or_path)

    # Use GPU if available
    # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = 'cpu'
    model.to(device_name)
    print(f'Model loaded on {device_name.upper()}')

    return model, processor, device_name


def run_inference(image_path: str, model, processor, device_name: str):
    """
    Runs inference on a single image.
    """
    # Load Image
    image = Image.open(image_path)
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


def visualize_result(image, result: dict, model, confidence_threshold: float = 0.5, instance_mode: bool = True) -> None:
    """
    Visualizes the instance segmentation masks overlaid on the image.

    Args:
        image: The input image.
        result: The segmentation result dictionary.
        model: The model used for inference (needed for label mapping).
        confidence_threshold: Minimum score to display a segment.
        instance_mode: If True, assigns a unique color and legend entry to every individual instance (e.g., "Car 1", "Car 2").
                       If False, assigns colors and legend entries by class (e.g., all "Car" instances are red).
    """
    segmentation = result['segmentation'].cpu().numpy()
    segments_info = result['segments_info']

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    # Create an empty colored mask (RGBA for transparency)
    color_mask = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    legend_patches = []
    contours_to_draw = []

    # Track counts for each class label to assign indices
    class_counts = {}

    # Track classes already added to the legend (for instance_mode=False)
    seen_classes_in_legend = set()

    # Filter segments first
    valid_segments = [s for s in segments_info if s['score'] >= confidence_threshold]
    num_instances = len(valid_segments)

    print(f'Found {len(segments_info)} raw instances, {num_instances} passed threshold.')

    # --- COLOR GENERATION LOGIC ---
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
        color_palette = [cmap(i) for i in range(num_colors_needed)]
    else:
        cmap = plt.get_cmap('nipy_spectral')
        color_palette = [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]

    # --- MAIN LOOP ---
    for i, segment in enumerate(valid_segments):
        segment_id = segment['id']
        label_id = segment['label_id']
        label_text = model.config.id2label[label_id]

        # Update count for this class (used for naming in instance_mode)
        count = class_counts.get(label_text, 0) + 1
        class_counts[label_text] = count

        # --- DETERMINE COLOR AND LABEL ---
        if instance_mode:
            # 1. Instance Mode: Unique color per instance, Label includes count
            rgb = color_palette[i][:3]
            display_label = f"{label_text} {count}"
            should_add_to_legend = True
        else:
            # 2. Class Mode: Color based on class ID, Label is just class name
            color_idx = class_color_index_map[label_id]
            rgb = color_palette[color_idx][:3]
            display_label = label_text

            # Only add to legend if we haven't seen this class yet
            if label_id not in seen_classes_in_legend:
                should_add_to_legend = True
                seen_classes_in_legend.add(label_id)
            else:
                should_add_to_legend = False

        # --- DRAWING ---

        # 1. High Transparency Cover (Low Alpha)
        fill_alpha = 0.3
        fill_color = np.concatenate([rgb, [fill_alpha]])

        # Apply fill color to the mask location
        mask_bool = (segmentation == segment_id)
        color_mask[mask_bool] = fill_color

        # Store contour data
        # 2. Low Transparency Contour (High Alpha)
        contour_alpha = 0.9
        contour_color = np.concatenate([rgb, [contour_alpha]])
        contours_to_draw.append((mask_bool, contour_color))

        # Add to legend if criteria met
        if should_add_to_legend:
            patch = mpatches.Patch(color=fill_color, label=display_label)
            legend_patches.append(patch)

    # Show the transparent fills
    ax.imshow(color_mask)

    # Draw contours on top of the fills
    for mask, color in contours_to_draw:
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)

    ax.axis('off')

    # Add a legend to the plot
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 1))

    # Update title based on mode
    mode_str = "Instance" if instance_mode else "Class"
    plt.title(f'Mask2Former (Swin-L) {mode_str} Segmentation', fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load Model
    model, processor, device_name = load_model()

    # Run Pipeline
    DATA_FOLDER_PATH = '../data/'
    # IMAGE_NAME = '20230607_095156.jpg'
    # IMAGE_NAME = 'IMG_20230505_164625.jpg'
    IMAGE_NAME = 'TestSorghumWeed (7).JPG'
    # IMAGE_NAME = 'TestSorghumWeed (14).JPG'
    image, result = run_inference(
        image_path=DATA_FOLDER_PATH + IMAGE_NAME,
        model=model,
        processor=processor,
        device_name=device_name,
    )

    # Show Output
    visualize_result(image=image, result=result, model=model, instance_mode=False)
