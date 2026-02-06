import os
import torch
import warnings
# import argparse
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

import config
from data_utils import WeedDataset, collate_fn
from evaluation_utils import test_with_metrics, print_metrics_evaluation

# Suppress specific warning about unused arguments in preprocessor config
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    message='.*The following named arguments are not valid.*'
)


def main(args):
    """Main function to run the testing."""
    # run_dir = args.model_run_dir
    run_dir = 'models/mask2former_fine_tuned/2026-02-04_21-46-40/'
    if not os.path.isdir(run_dir):
        print(f'Error: Directory not found at "{run_dir}"')
        return

    print(f'--- Starting Testing for run: {os.path.basename(run_dir)} ---')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    final_model_path = os.path.join(run_dir, 'final_model')
    best_model_path = os.path.join(run_dir, 'best_model')

    processor_path = final_model_path if os.path.exists(final_model_path) else best_model_path
    if not os.path.exists(processor_path):
        print(f'Error: No model or processor found in "{run_dir}". Searched for "final_model" and "best_model".')
        return

    print(f'Loading processor from: {processor_path}')
    processor = AutoImageProcessor.from_pretrained(processor_path, use_fast=False)

    print('\nLoading test dataset...')
    test_dataset = WeedDataset(config.TEST_IMG_DIR, config.TEST_JSON, processor)
    if len(test_dataset) == 0:
        print('No test data found. Skipping testing.')
        return
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if os.path.exists(final_model_path):
        print(f'\nTesting final model from: {final_model_path}')
        model = Mask2FormerForUniversalSegmentation.from_pretrained(final_model_path).to(device)
        metrics_evaluation = test_with_metrics(model, processor, test_loader, device)
        print_metrics_evaluation(metrics_evaluation=metrics_evaluation, model_name='Final Model')
    else:
        print('\n"final_model" not found. Skipping.')

    if os.path.exists(best_model_path):
        print(f'\nTesting best model from: {best_model_path}')
        model = Mask2FormerForUniversalSegmentation.from_pretrained(best_model_path).to(device)
        metrics_evaluation = test_with_metrics(model, processor, test_loader, device)
        print_metrics_evaluation(metrics_evaluation=metrics_evaluation, model_name='Best Model')
    else:
        print('\n"best_model" not found. Skipping.')

    print('\n--- Testing Complete ---')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Test a fine-tuned Mask2Former model.')
    # parser.add_argument(
    #     'model_run_dir',
    #     type=str,
    #     help='Path to the specific training run directory (e.g., "models/mask2former_fine_tuned/YYYY-MM-DD_HH-MM-SS")'
    # )
    # cli_args = parser.parse_args()
    cli_args = None
    main(cli_args)
