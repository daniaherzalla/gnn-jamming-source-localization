import csv
import json
import os
from typing import Dict
import numpy as np
import torch


def save_metrics_and_params(metrics: Dict[str, float], params: Dict[str, float], filename: str = 'model_metrics_and_params.csv') -> None:
    """
    Save metrics and parameters to a JSON file.

    Args:
        metrics (Dict[str, float]): Dictionary of metrics.
        params (Dict[str, float]): Dictionary of parameters.
        filename (str): Filename for the JSON file. Default is 'model_metrics_and_params.json'.
    """
    # Remove certain keys from params
    [params.pop(key, None) for key in ['dataset_path', 'train_path', 'val_path', 'test_path']]

    # Create a dictionary to store both metrics and params
    result = {**metrics, **params}

    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())

        # Write the header only if the file didn't exist before
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(result)


def set_random_seeds(seed_value=42):
    np.random.seed(seed_value + 1)
    torch.manual_seed(seed_value + 2)
    torch.cuda.manual_seed_all(seed_value + 3)


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
