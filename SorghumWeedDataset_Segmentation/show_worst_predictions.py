import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

import config
from data_utils import PreprocessedWeedDataset, WeedDataset, collate_fn
from Mask2Former_inference import load_model, run_inference, load_ground_truth, visualize_result

N_WORST = 3  # Number of worst cases to show
MODEL_ID_OR_PATH = 'models/mask2former_fine_tuned/2026-02-06_02-49-29/best_model'


def get_batch_targets(original_maps, id_mappings) -> list:
    """
    Reconstructs target masks/labels from the original instance map
    in the format expected by torchmetrics.
    """
    targets = []
    for k in range(len(original_maps)):
        gt_map = original_maps[k]
        mapping = id_mappings[k]

        masks = []
        labels = []

        # Reconstruct binary masks from the instance map
        unique_ids = np.unique(gt_map)
        for uid in unique_ids:
            # Skip 255 (background)
            if uid == 255:
                continue
            if uid not in mapping:
                continue

            # Create binary mask for this instance
            bin_mask = torch.from_numpy((gt_map == uid)).bool()
            class_id = mapping[uid]

            masks.append(bin_mask)
            labels.append(class_id)

        if len(masks) > 0:
            targets.append({
                'masks': torch.stack(masks).to('cpu'),
                'labels': torch.tensor(labels).to('cpu')
            })
        else:
            # Handle images with no valid objects
            targets.append({
                'masks': torch.zeros((0, *gt_map.shape), dtype=torch.bool),
                'labels': torch.tensor([], dtype=torch.long)
            })
    return targets


def get_batch_predictions(outputs, processor, target_sizes) -> list:
    """
    Post-processes model outputs to getting predictions in format expected by torchmetrics.
    """
    predictions = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=target_sizes,
        threshold=0.5,
        mask_threshold=0.5
    )

    formatted_predictions = []
    for pred in predictions:
        segments_info = pred['segments_info']

        if not segments_info:
            formatted_predictions.append({
                'masks': torch.empty(0, *pred['segmentation'].shape, dtype=torch.bool, device='cpu'),
                'scores': torch.empty(0, device='cpu'),
                'labels': torch.empty(0, dtype=torch.long, device='cpu'),
            })
            continue

        scores = torch.tensor([info['score'] for info in segments_info])
        labels = torch.tensor([info['label_id'] for info in segments_info])

        instance_map = pred['segmentation']
        instance_ids = [info['id'] for info in segments_info]
        masks = torch.stack([instance_map == iid for iid in instance_ids])

        formatted_predictions.append({
            'masks': masks.cpu(),
            'scores': scores.cpu(),
            'labels': labels.cpu(),
        })
    return formatted_predictions


def main(model_id_or_path, n_worst: int = 5):
    # 1. Load Model
    model, processor, device_name = load_model(model_id_or_path)
    device = torch.device(device_name)
    model.eval()

    # 2. Prepare Dataset
    # Try to load preprocessed test data first, else fallback to raw
    test_processed_path = os.path.join(config.PROCESSED_DIR, 'Test')
    if os.path.exists(test_processed_path) and len(os.listdir(test_processed_path)) > 0:
        print(f"Loading pre-processed test data from {test_processed_path}")
        test_dataset = PreprocessedWeedDataset(test_processed_path)
    else:
        print("Loading raw test data (this might be slower)...")
        test_dataset = WeedDataset(config.TEST_IMG_DIR, config.TEST_JSON, processor)

    if len(test_dataset) == 0:
        print("No test data found.")
        return

    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 3. Evaluation Loop
    scored_images = []  # List of tuples: (map_score, filename)
    metric = MeanAveragePrecision(iou_type='segm')

    print(f"\nEvaluating {len(test_dataset)} images to find worst predictions...")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if (i + 1) % 5 == 0:
                print(f"  Processing image {i + 1}/{len(test_dataset)}...", end='\r')

            pixel_values = batch['pixel_values'].to(device)
            target_sizes = batch['target_sizes']
            file_names = batch['file_names']

            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Prepare data for metric calculation
            formatted_preds = get_batch_predictions(outputs, processor, target_sizes)
            targets = get_batch_targets(batch['original_maps'], batch['id_mappings'])

            # batch_size=1 to pair score with filename easily
            for j in range(len(formatted_preds)):
                # Reset metric for single image evaluation
                metric.reset()
                metric.update([formatted_preds[j]], [targets[j]])
                result = metric.compute()

                # 'map' is Mean Average Precision. Lower is worse.
                # If map is -1 (tensor(-1)), it usually means undefined/no ground truth,
                # but torchmetrics usually returns 0.0 or valid number.
                score = result['map'].item()

                scored_images.append((score, file_names[j]))

    # 4. Sort and Select Worst
    # Sort by score ascending (lowest score first)
    scored_images.sort(key=lambda x: x[0])

    worst_cases = scored_images[:n_worst]

    print(f"\n\n--- Top {n_worst} Worst Predictions (by mAP) ---")
    for score, file_name in worst_cases:
        print(f"File: {file_name} | mAP: {score:.4f}")

    # 5. Visualize
    print("\nVisualizing Worst Cases...")
    for score, file_name in worst_cases:
        print(f"\nVisualizing: {file_name} (mAP: {score:.4f})")

        image_path = os.path.join(config.TEST_IMG_DIR, file_name)
        if not os.path.exists(image_path):
            print(f"Could not find original image at {image_path}, skipping visualization.")
            continue

        # Run inference using the pipeline from Mask2Former_inference
        # This re-runs inference, which is slightly inefficient but ensures
        # the visualization logic matches exactly what was requested.
        image, result = run_inference(
            image_path=image_path,
            model=model,
            processor=processor,
            device_name=device_name
        )

        gt_result = load_ground_truth(file_name, image.size)

        visualize_result(
            image=image,
            result=result,
            model=model,
            ground_truth_result=gt_result,
            confidence_threshold=0.5,
            instance_mode=False
        )


if __name__ == '__main__':
    main(MODEL_ID_OR_PATH, N_WORST)
