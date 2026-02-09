import os
import glob
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Generic loader for pre-processed .pt files from disk.
    Expected structure: list of .pt files containing dictionaries.
    """
    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir
        self.files = glob.glob(os.path.join(processed_dir, "*.pt"))
        self.files.sort()

        if len(self.files) == 0:
            print(f"WARNING: No .pt files found in {processed_dir}. Has preprocessing been run?")
        else:
            print(f"Loaded {len(self.files)} pre-processed samples from {processed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Load the dictionary directly from the .pt file
        data = torch.load(file_path, weights_only=False)
        return data


def collate_fn(batch) -> dict:
    """
    Collates a batch of dictionaries into a dictionary of stacked tensors/lists.
    Compatible with Mask2Former inputs.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    mask_labels = [item['mask_labels'] for item in batch]
    class_labels = [item['class_labels'] for item in batch]
    target_sizes = [item['target_size'] for item in batch]
    original_maps = [item['original_map'] for item in batch]
    id_mappings = [item['id_to_semantic'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'mask_labels': mask_labels,
        'class_labels': class_labels,
        'target_sizes': target_sizes,
        'original_maps': original_maps,
        'id_mappings': id_mappings,
        'file_names': file_names,
    }
