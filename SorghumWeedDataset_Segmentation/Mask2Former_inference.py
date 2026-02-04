import torch
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
    print('Loading Mask2Former (Swin-Large-COCO)... this may take a moment.')

    # 'facebook/mask2former-swin-large-coco-instance' is the high-accuracy
    # checkpoint fine-tuned for instance segmentation.
    model_id = 'facebook/mask2former-swin-large-coco-instance'

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'Model loaded on {device.upper()}')

    return model, processor, device


def run_inference(image_path: str, model, processor, device: str):
    """
    Runs inference on a single image.
    """
    # Load Image
    image = Image.open(image_path)

    print(f'Processing image size: {image.size}')

    # Preprocess
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

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


def visualize_result(image, result: dict, model, confidence_threshold: float = 0.5) -> None:
    """
    Visualizes the instance segmentation masks overlaid on the image.
    """
    segmentation = result['segmentation'].cpu().numpy()
    segments_info = result['segments_info']

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    # Create an empty colored mask
    # RGBA for transparency
    color_mask = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    legend_patches = []
    contours_to_draw = []

    # Track counts for each class label to assign indices
    class_counts = {}

    # Filter segments first to determine the exact number of colors needed
    valid_segments = [s for s in segments_info if s['score'] >= confidence_threshold]
    num_instances = len(valid_segments)

    print(f'Found {len(segments_info)} raw instances, {num_instances} passed threshold.')

    # Generate distinct colors based on the number of instances
    # 'tab20' is a qualitative colormap with 20 distinct colors.
    # 'nipy_spectral' is a high-contrast continuous map good for many classes.
    if num_instances <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(num_instances)]
    else:
        cmap = plt.get_cmap('nipy_spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, num_instances)]

    # Loop through filtered segments
    for i, segment in enumerate(valid_segments):
        segment_id = segment['id']
        label_id = segment['label_id']
        # score check already done during filtering

        label_text = model.config.id2label[label_id]

        # Update count for this class
        count = class_counts.get(label_text, 0) + 1
        class_counts[label_text] = count

        # Create display label with index
        display_label = f"{label_text} {count}"

        # Get the assigned distinct color (slice :3 to get RGB, ignore alpha from cmap)
        rgb = colors[i][:3]

        # 1. High Transparency Cover (Low Alpha)
        fill_alpha = 0.3
        fill_color = np.concatenate([rgb, [fill_alpha]])

        # Apply fill color to the mask location
        mask_bool = (segmentation == segment_id)
        color_mask[mask_bool] = fill_color

        # Store contour data to draw later (to ensure they sit on top)
        # 2. Low Transparency Contour (High Alpha)
        contour_alpha = 0.9
        contour_color = np.concatenate([rgb, [contour_alpha]])
        contours_to_draw.append((mask_bool, contour_color))

        # Add to legend for every instance
        patch = mpatches.Patch(color=fill_color, label=display_label)
        legend_patches.append(patch)

    # Show the transparent fills
    ax.imshow(color_mask)

    # Draw contours on top of the fills
    for mask, color in contours_to_draw:
        # levels=[0.5] draws the boundary for the boolean mask
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)

    ax.axis('off')

    # Add a legend to the plot
    # Adjusted bbox_to_anchor to account for potentially longer legend list
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.title(f'Mask2Former (Swin-L) Instance Segmentation', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load Model
    model, processor, device = load_model()

    # Run Pipeline
    DATA_FOLDER_PATH = '../data/'
    # IMAGE_NAME = '20230607_095156.jpg'
    IMAGE_NAME = 'IMG_20230505_164625.jpg'
    image, result = run_inference(
        image_path=DATA_FOLDER_PATH + IMAGE_NAME,
        model=model,
        processor=processor,
        device=device,
    )

    # Show Output
    visualize_result(image=image, result=result, model=model)
