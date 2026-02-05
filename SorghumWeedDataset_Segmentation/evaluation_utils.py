import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision


def test_with_metrics(model, processor, data_loader, device):
    """
    Evaluates the model on the test set using detailed metrics (mAP).
    """
    model.eval()
    map_metric = MeanAveragePrecision(iou_type='segm')

    for batch in tqdm(data_loader, desc='Calculating Metrics'):
        pixel_values = batch['pixel_values'].to(device)
        target_sizes = batch['target_sizes']

        targets = []
        for i in range(len(pixel_values)):
            targets.append({
                'masks': batch['mask_labels'][i].to(torch.bool),
                'labels': batch['class_labels'][i],
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

    print(f'  mAP:              {metrics_evaluation.get("map", torch.tensor(-1)).item():.4f}')
    print(f'  mAP (IoU=0.50):   {metrics_evaluation.get("map_50", torch.tensor(-1)).item():.4f}')
    print(f'  mAP (IoU=0.75):   {metrics_evaluation.get("map_75", torch.tensor(-1)).item():.4f}')
    print('-' * 25)
    print(f'  mAP (small):      {metrics_evaluation.get("map_small", torch.tensor(-1)).item():.4f}')
    print(f'  mAP (medium):     {metrics_evaluation.get("map_medium", torch.tensor(-1)).item():.4f}')
    print(f'  mAP (large):      {metrics_evaluation.get("map_large", torch.tensor(-1)).item():.4f}')


def prepare_metrics_for_json(metrics):
    """Converts tensors in metrics dict to floats for JSON serialization."""
    if not metrics:
        return None
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
