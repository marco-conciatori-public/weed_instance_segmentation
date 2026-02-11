import importlib


def get_dataset_and_config(dataset_name: str):
    """
    Dynamically imports and returns the configuration module and specific Dataset class for the given dataset name.
    """
    config_module_path = 'datasets.' + dataset_name + '.definitions'
    try:
        config_module = importlib.import_module(config_module_path)
    except Exception:
        raise ValueError(f'config_module for dataset {dataset_name} not found.'
                         f' (check path "ROOT/{config_module_path}")')

    dataset_module_path = 'datasets.' + dataset_name + '.dataset'
    try:
        dataset_module = importlib.import_module(dataset_module_path)
    except Exception:
        raise ValueError(f'dataset_module for dataset {dataset_name} not found.'
                         f' (check path "ROOT/{dataset_module_path}")')

    dataset_class_name = dataset_name.title().replace('_', '') + 'Dataset'
    try:
        dataset_class = getattr(dataset_module, dataset_class_name)
    except Exception:
        raise ValueError(f'dataset_class for dataset {dataset_name} not found.'
                         f' (check class "ROOT/{dataset_module_path}.{dataset_class_name}")')

    return dataset_class, config_module
