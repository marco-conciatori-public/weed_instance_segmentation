import torch
import numpy as np
from torchmetrics.detection import MeanAveragePrecision


def test_with_metrics(model, processor, data_loader, device):
    """
    Evaluates the model on the test set using detailed metrics (mAP).
    """
    model.eval()
    map_metric = MeanAveragePrecision(iou_type='segm')

    print('Calculating Metrics...')
    for i, batch in enumerate(data_loader):
        if (i + 1) % 5 == 0:
            print(f'  Processing batch {i + 1}/{len(data_loader)}')

        pixel_values = batch['pixel_values'].to(device)
        target_sizes = batch['target_sizes']

        # Construct targets from original maps to ensure precise shape matching
        # (Processor pads images, so mask_labels are padded. Post-process crops predictions.
        # Targets need to be unpadded to match predictions.)
        targets = []
        original_maps = batch['original_maps']
        id_mappings = batch['id_mappings']

        for k in range(len(pixel_values)):
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

                # Safety check
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

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

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

        map_metric.update(formatted_predictions, targets)

    results = map_metric.compute()
    model.train()  # Set model back to training mode for consistency if training continues
    return results


def print_metrics_evaluation(metrics_evaluation, model_name: str = 'Model'):
    """Helper function to print metrics from torchmetrics."""
    print(f'\n--- {model_name} Metrics ---')
    if not metrics_evaluation:
        print('No metrics calculated.')
        return

    # Helper to safely get scalar item
    def get_scalar(key):
        val = metrics_evaluation.get(key, torch.tensor(-1))
        # Ensure we only call .item() on scalars
        return val.item() if val.numel() == 1 else -1

    print(f'  mAP:              {get_scalar("map"):.4f}')
    print(f'  mAP (IoU=0.50):   {get_scalar("map_50"):.4f}')
    print(f'  mAP (IoU=0.75):   {get_scalar("map_75"):.4f}')
    print('-' * 25)
    print(f'  mAP (small):      {get_scalar("map_small"):.4f}')
    print(f'  mAP (medium):     {get_scalar("map_medium"):.4f}')
    print(f'  mAP (large):      {get_scalar("map_large"):.4f}')


def prepare_metrics_for_json(metrics):
    """Converts tensors in metrics dict to floats/lists for JSON serialization."""
    if not metrics:
        return None

    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            # If scalar, convert to float (e.g. mAP)
            if v.numel() == 1:
                clean_metrics[k] = v.item()
            # If vector/array, convert to list (e.g. classes tensor)
            else:
                clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    return clean_metrics
