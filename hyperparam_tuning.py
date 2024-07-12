import json
import torch
import random
from hyperopt import hp, fmin, tpe, Trials, space_eval
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader
from config import params
import numpy as np
# print(torch.cuda.is_available())
# quit()

def objective(hyperparameters):
    """
    Objective function for hyperparameter optimization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    print("hyperparameters: ", hyperparameters)
    train_dataset, val_dataset, test_dataset, original_dataset = load_data(params['dataset_path'], hyperparameters)
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=hyperparameters['batch_size'])
    steps_per_epoch = len(train_loader)
    model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters, steps_per_epoch)
    for epoch in range(params['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')
    return val_loss


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


def map_indices_to_values(hyperparameters):
    """
    Map indices to actual values for hyperparameters that use hp.choice.
    """
    actual_values = {
        "num_heads": [2, 4, 8],
        "num_layers": [2, 4, 8],
        "hidden_channels": [32, 64, 128, 256],
        "out_channels": [32, 64, 128, 256],
        "batch_size": [16, 32, 64],
        "max_epochs": [200],
        "num_neighbors": [5, 10, 20, 30],
        'additional_features': additional_features_evaluated,
        'required_features': ['node_positions', 'node_noise'],
        "coords": ['cartesian'],
        "edges": ['knn'],
        "norm": ['minmax'],
        "model": ['GATv2']
    }
    return {key: actual_values[key][hyperparameters[key][0]] if key in actual_values and hyperparameters[key] else hyperparameters[key][0] if hyperparameters[key] else None for key in hyperparameters}


def save_results(trials, best_hyperparams, best_loss):
    """
    Save the results of the hyperparameter tuning process along with the best validation loss.
    """
    results = {
        'best_hyperparameters': convert_to_serializable(best_hyperparams),
        'best_loss': convert_to_serializable(best_loss),
        'trials': [
            {
                'hyperparameters': convert_to_serializable(map_indices_to_values(trial['misc']['vals'])),
                'result': convert_to_serializable(trial['result'])
            } for trial in trials.trials
        ]
    }
    model_name = best_hyperparams['model']
    filename = f'hyperparam_results/trial_results_{model_name}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def sample_additional_features():
    """
    Sample additional features randomly. Ensure that if 'sin_azimuth' or 'cos_azimuth' is selected,
    the other is also included if not already present.
    """
    features = ['dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise']
    num_features = random.randint(1, len(features))
    sampled_features = random.sample(features, num_features)

    # Check if 'sin_azimuth' is in the sampled features and 'cos_azimuth' is not, and vice versa
    if 'sin_azimuth' in sampled_features and 'cos_azimuth' not in sampled_features:
        sampled_features.append('cos_azimuth')
    elif 'cos_azimuth' in sampled_features and 'sin_azimuth' not in sampled_features:
        sampled_features.append('sin_azimuth')

    return sampled_features


def main():
    """
    Main function to run hyperparameter optimization.
    """
    # Pre-evaluate additional features
    global additional_features_evaluated
    additional_features_evaluated = sample_additional_features()
    hyperparameter_space = {
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.6),
        'num_heads': hp.choice('num_heads', [2, 4, 8]),
        'num_layers': hp.choice('num_layers', [2, 4, 8]),
        'hidden_channels': hp.choice('hidden_channels', [32, 64, 128, 256]),
        'out_channels': hp.choice('out_channels', [32, 64, 128, 256]),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'weight_decay': hp.loguniform('weight_decay', -6, -1),
        'num_neighbors': hp.choice('num_neighbors', [5, 10, 20, 30]),
        'additional_features': additional_features_evaluated,
        'required_features': ['node_positions', 'node_noise'],
        'max_epochs': hp.choice('max_epochs', [200]),
        'coords': hp.choice('coords', ['cartesian']),
        'edges': hp.choice('edges', ['knn']),
        'norm': hp.choice('norm', ['minmax']),
        'model': hp.choice('model', ['GATv2'])
    }
    trials = Trials()
    best_hyperparameters = fmin(
        fn=lambda hyperparameters: objective(hyperparameters),
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    best_hyperparams = space_eval(hyperparameter_space, best_hyperparameters)
    best_loss = min([trial['result']['loss'] for trial in trials.trials if trial['result']['status'] == 'ok'])
    save_results(trials, best_hyperparams, best_loss)


if __name__ == "__main__":
    main()
