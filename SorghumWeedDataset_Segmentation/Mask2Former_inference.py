import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor


def load_model():
    '''
    Loads the Mask2Former model and processor
    with the Swin-Large backbone for maximum accuracy
    '''
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
    '''
    Runs inference on a single image.
    '''
    # Load Image
    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    print(f'Processing image size: {image.size}')

    # Preprocess
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    # We must provide target_sizes (height, width) to resize masks back to original image
    result = processor.post_process_instance_segmentation(
        outputs=outputs,
        target_sizes=[image.size[::-1]]  # (Height, Width)
    )[0]

    return image, result


def visualize_result(image, result: dict, model, confidence_threshold: float = 0.5) -> None:
    '''
    Visualizes the instance segmentation masks overlaid on the image.
    '''
    segmentation = result['segmentation'].cpu().numpy()
    segments_info = result['segments_info']

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    # Create an empty colored mask
    # We use RGBA for transparency
    color_mask = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    legend_patches = []
    seen_labels = set()

    print(f'Found {len(segments_info)} instances.')

    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        score = segment['score']

        # Filter low confidence predictions if desired
        if score < confidence_threshold:
            continue

        label_text = model.config.id2label[label_id]

        # Generate a random vivid color for this instance
        color = np.concatenate([np.random.random(3), [0.6]])  # 0.6 is alpha (transparency)

        # Apply color to the mask location
        mask_bool = (segmentation == segment_id)
        color_mask[mask_bool] = color

        # Add to legend if we haven't seen this class yet (or for every instance)
        # Here we add a legend entry for every unique class found
        if label_id not in seen_labels:
            patch = mpatches.Patch(color=color, label=f'{label_text}')
            legend_patches.append(patch)
            seen_labels.add(label_id)

    ax.imshow(color_mask)
    ax.axis('off')

    # Add a legend to the plot
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title(f'Mask2Former (Swin-L) Instance Segmentation', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Load Model
    model, processor, device = load_model()

    # 2. Define Image (URL or local path)
    # Using a standard COCO validation image (Street scene)
    # IMAGE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    IMAGE_URL = '../data/20230607_095156.jpg'

    # 3. Run Pipeline
    image, result = run_inference(IMAGE_URL, model, processor, device)

    # 4. Show Output
    visualize_result(image, result, model)
