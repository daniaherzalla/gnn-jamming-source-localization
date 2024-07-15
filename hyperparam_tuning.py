import json
import torch
import random
from hyperopt import hp, fmin, tpe, Trials, space_eval
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader
from config import params
import numpy as np


def objective(hyperparameters):
    """
    Objective function for hyperparameter optimization.
    """
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    print("hyperparameters: ", hyperparameters)

    # Decide which additional features to include based on hyperparameters
    additional_features = []
    feature_names = [
        'dist_to_centroid', 'relative_noise', 'proximity_count',
        'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'
    ]
    # Handle azimuth features together
    if hyperparameters['use_sin_azimuth'] or hyperparameters['use_cos_azimuth']:
        additional_features.extend(['sin_azimuth', 'cos_azimuth'])

    # Add other features if enabled
    for feature in feature_names:
        if hyperparameters[f'use_{feature}']:
            additional_features.append(feature)

    # Include the additional features in the hyperparameters dictionary
    hyperparameters['additional_features'] = additional_features

    train_dataset, val_dataset, test_dataset, original_dataset = load_data(params['dataset_path'], hyperparameters)
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=hyperparameters['batch_size'])
    steps_per_epoch = len(train_loader)
    model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters, steps_per_epoch)
    for epoch in range(params['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')

    # Return the results with the final hyperparameters including the dynamically added additional features
    return {'loss': val_loss, 'status': 'ok'}


def convert_to_serializable(data):
    """
    Convert numpy data types to Python native types for JSON serialization.
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    else:
        return data


def map_indices_to_values(hyperparameters, model_type):
    """
    Map indices to actual values for hyperparameters that use hp.choice.
    Include additional_features dynamically sampled in each trial.
    """
    actual_values = {
        "num_heads": [2, 4, 8],
        "num_layers": [2, 4, 8],
        "hidden_channels": [32, 64, 128, 256],
        "out_channels": [32, 64, 128, 256],
        "batch_size": [16, 32, 64],
        "max_epochs": [200],
        "num_neighbors": [5, 10, 20, 30],
        "coords": ['cartesian'],
        "edges": ['knn'],
        "norm": ['minmax'],
        "model": [model_type]
    }

    mapped_hyperparameters = {
        key: actual_values[key][hyperparameters[key][0]] if key in actual_values and isinstance(hyperparameters[key], list) and hyperparameters[key] else hyperparameters[key]
        for key in hyperparameters
    }
    mapped_hyperparameters['required_features'] = ['node_positions', 'node_noise']

    return mapped_hyperparameters


def save_results(trials, best_hyperparams, best_loss, model_type):
    """
    Save the results of the hyperparameter tuning process along with the best validation loss.
    Adjust the function to save additional_features within the hyperparameters section.
    """
    results = {
        'best_hyperparameters': convert_to_serializable(map_indices_to_values(best_hyperparams, model_type)),
        'best_loss': convert_to_serializable(best_loss),
        'trials': [
            {
                'hyperparameters': convert_to_serializable(map_indices_to_values(trial['misc']['vals'], model_type)),
                'result': convert_to_serializable({
                    'loss': trial['result']['loss'],
                    'status': trial['result']['status']
                }),
            } for trial in trials.trials if trial['result']['status'] == 'ok'
        ]
    }
    model_name = best_hyperparams['model']
    filename = f'hyperparam_results/trial_results_{model_name}_TRIAL2.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    """
    Main function to run hyperparameter optimization.
    """
    model_type = 'MLP'

    hyperparameter_space = {
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.6),
        'num_heads': hp.choice('num_heads', [2, 4, 8]),
        'num_layers': hp.choice('num_layers', [2, 4, 8]),
        'hidden_channels': hp.choice('hidden_channels', [32, 64, 128, 256]),
        'out_channels': hp.choice('out_channels', [32, 64, 128, 256]),
        'batch_size': hp.choice('batch_size', [16, 32]),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.001),
        'weight_decay': hp.loguniform('weight_decay', -7, -2),
        'num_neighbors': hp.choice('num_neighbors', [5, 10, 20, 30]),
        'required_features': ['node_positions', 'node_noise'],
        'max_epochs': hp.choice('max_epochs', [200]),
        'coords': hp.choice('coords', ['cartesian']),
        'edges': hp.choice('edges', ['knn']),
        'norm': hp.choice('norm', ['minmax']),
        'model': hp.choice('model', [model_type]),
        # Additional features included as binary choices
        'use_dist_to_centroid': hp.choice('use_dist_to_centroid', [True, False]),
        'use_sin_azimuth': hp.choice('use_sin_azimuth', [True, False]),
        'use_cos_azimuth': hp.choice('use_cos_azimuth', [True, False]),
        'use_relative_noise': hp.choice('use_relative_noise', [True, False]),
        'use_proximity_count': hp.choice('use_proximity_count', [True, False]),
        'use_clustering_coefficient': hp.choice('use_clustering_coefficient', [True, False]),
        'use_mean_noise': hp.choice('use_mean_noise', [True, False]),
        'use_median_noise': hp.choice('use_median_noise', [True, False]),
        'use_std_noise': hp.choice('use_std_noise', [True, False]),
        'use_range_noise': hp.choice('use_range_noise', [True, False])
    }

    trials = Trials()
    best_hyperparameters = fmin(
        fn=objective,
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )
    best_hyperparams = space_eval(hyperparameter_space, best_hyperparameters)
    best_loss = min([trial['result']['loss'] for trial in trials.trials if trial['result']['status'] == 'ok'])
    save_results(trials, best_hyperparams, best_loss, model_type)


if __name__ == "__main__":
    main()