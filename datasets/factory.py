import importlib

# Register your datasets here
DATASET_REGISTRY = {
    "sorghum_weed": "datasets.sorghum_weed.definitions",
    # Future datasets:
    # "corn_weed": "datasets.corn_weed.definitions",
}


def get_dataset_config(dataset_name: str):
    """
    Dynamically imports and returns the configuration module for the given dataset name.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_REGISTRY.keys())}")

    module_path = DATASET_REGISTRY[dataset_name]
    config_module = importlib.import_module(module_path)
    return config_module
