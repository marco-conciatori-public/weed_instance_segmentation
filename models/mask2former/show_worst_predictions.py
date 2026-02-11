import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

import config
from datasets.factory import get_dataset_and_config
from models.mask2former.inference import run_inference
from models.model_utils import load_model, plot_segmentation
from datasets.dataset_utils import PreprocessedDataset, collate_fn

N_WORST = 3
MODEL_ID = 'mask2former_fine_tuned/2026-02-09_19-50-56/best_model/'


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

        unique_ids = np.unique(gt_map)
        for uid in unique_ids:
            if uid == 255:
                continue
            if uid not in mapping:
                continue

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
            targets.append({
                'masks': torch.zeros((0, *gt_map.shape), dtype=torch.bool),
                'labels': torch.tensor([], dtype=torch.long)
            })
    return targets


def get_batch_predictions(outputs, processor, target_sizes) -> list:
    """
    Post-processes model outputs for torchmetrics.
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


def convert_gt_map_to_result(gt_map, id_mapping) -> dict:
    """
    Helper to convert the dataloader's original map into the dict format
    expected by plot_segmentation.
    """
    segments_info = []
    unique_ids = np.unique(gt_map)

    for uid in unique_ids:
        if uid == 255:
            continue
        if uid not in id_mapping:
            continue

        segments_info.append({
            'id': int(uid),
            'label_id': id_mapping[uid],
            'score': 1.0  # GT is always 100%
        })

    return {
        'segmentation': torch.from_numpy(gt_map),
        'segments_info': segments_info
    }


def main(model_id, n_worst: int = N_WORST):
    # 1. Load Model
    model, processor, device_name = load_model(model_id)
    device = torch.device(device_name)
    model.eval()

    WeedDataset, ds_config = get_dataset_and_config(config.DATASET_LIST[0])

    # 2. Prepare Dataset
    test_processed_path = os.path.join(ds_config.PROCESSED_DIR, 'Test')
    if os.path.exists(test_processed_path) and len(os.listdir(test_processed_path)) > 0:
        print(f"Loading pre-processed test data from {test_processed_path}")
        test_dataset = PreprocessedDataset(test_processed_path)
    else:
        print("Loading raw test data...")
        test_dataset = WeedDataset(
            image_folder_path=ds_config.TEST_IMG_DIR,
            annotation_file_path=ds_config.TEST_ANNOTATIONS,
            processor=processor,
            label2id=ds_config.LABEL2ID,
        )

    if len(test_dataset) == 0:
        print("No test data found.")
        return

    data_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Important for per-image scoring
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 3. Evaluation Loop
    scored_images = []
    metric = MeanAveragePrecision(iou_type='segm')

    print(f"\nEvaluating {len(test_dataset)} images...")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if (i + 1) % 5 == 0:
                print(f"  Processing {i + 1}/{len(test_dataset)}...", end='\r')

            pixel_values = batch['pixel_values'].to(device)
            target_sizes = batch['target_sizes']
            file_names = batch['file_names']

            outputs = model(pixel_values=pixel_values)

            formatted_preds = get_batch_predictions(outputs, processor, target_sizes)
            targets = get_batch_targets(batch['original_maps'], batch['id_mappings'])

            # Calculate mAP for this single image
            metric.reset()
            metric.update(formatted_preds, targets)
            result = metric.compute()

            score = result['map'].item()

            # Store necessary data for visualization to avoid re-loading everything later if possible
            # We store the original map and mapping to reconstruct GT easily
            scored_images.append({
                'score': score,
                'file_name': file_names[0],
                'original_map': batch['original_maps'][0],
                'id_mapping': batch['id_mappings'][0]
            })

    # 4. Sort and Select Worst
    scored_images.sort(key=lambda x: x['score'])
    worst_cases = scored_images[:n_worst]

    print(f"\n\n--- Top {n_worst} Worst Predictions (by mAP) ---")
    for case in worst_cases:
        print(f"File: {case['file_name']} | mAP: {case['score']:.4f}")

    # 5. Visualize
    print("\nVisualizing...")
    for case in worst_cases:
        file_name = case['file_name']
        score = case['score']
        print(f"Visualizing: {file_name} (mAP: {score:.4f})")

        # Load original image for plotting
        img_path = os.path.join(ds_config.TEST_IMG_DIR, file_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Re-run inference for the visualization specific logic (resizing, plotting structure)
        # We could use the cached prediction, but run_inference handles resizing logic for display nicely
        image, result = run_inference(img_path, model, processor, device_name)

        # Construct GT Result
        gt_result = convert_gt_map_to_result(case['original_map'], case['id_mapping'])

        # Plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        plot_segmentation(axes[0], image, result, model, instance_mode=False, score_threshold=0.5)
        axes[0].set_title(f"Prediction (mAP: {score:.2f})")

        plot_segmentation(axes[1], image, gt_result, model, instance_mode=False)
        axes[1].set_title("Ground Truth")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main(MODEL_ID, N_WORST)
