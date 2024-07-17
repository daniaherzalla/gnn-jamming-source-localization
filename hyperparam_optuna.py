import json
import torch
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader
import numpy as np

import time


def objective(trial):
    """
    Objective function for hyperparameter optimization.
    Attempts to run the trial and returns infinity if an error occurs,
    signaling a failed trial.
    """

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    hyperparameters = {
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.7),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'num_layers': trial.suggest_categorical('num_layers', [2, 4, 8]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512]),
        'out_channels': trial.suggest_categorical('out_channels', [32, 64, 128, 256, 512]),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.001),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'num_neighbors': trial.suggest_categorical('num_neighbors', [5, 10, 20, 30]),
        'required_features': ['node_positions', 'node_noise'],
        'max_epochs': 200,
        'coords': 'cartesian',
        'edges': 'knn',
        'norm': 'minmax',
        'model': 'GCN',
        # Dynamic feature selection
        'use_dist_to_centroid': trial.suggest_categorical('use_dist_to_centroid', [True, False]),
        'use_sin_azimuth': trial.suggest_categorical('use_sin_azimuth', [True, False]),
        'use_relative_noise': trial.suggest_categorical('use_relative_noise', [True, False]),
        'use_proximity_count': trial.suggest_categorical('use_proximity_count', [True, False]),
        'use_clustering_coefficient': trial.suggest_categorical('use_clustering_coefficient', [True, False]),
        'use_mean_noise': trial.suggest_categorical('use_mean_noise', [True, False]),
        'use_median_noise': trial.suggest_categorical('use_median_noise', [True, False]),
        'use_std_noise': trial.suggest_categorical('use_std_noise', [True, False]),
        'use_range_noise': trial.suggest_categorical('use_range_noise', [True, False])
    }

    additional_features = [feature for feature in [
        'dist_to_centroid', 'relative_noise', 'proximity_count',
        'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'
    ] if hyperparameters[f'use_{feature}']]

    # Include azimuth features if selected
    if hyperparameters['use_sin_azimuth']:
        additional_features.extend(['sin_azimuth', 'cos_azimuth'])

    hyperparameters['additional_features'] = additional_features


    try:
        train_dataset, val_dataset, test_dataset, original_dataset = load_data('/home/dania/Downloads/dataset/random/random.csv', hyperparameters)
        train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=hyperparameters['batch_size'])
        steps_per_epoch = len(train_loader)
        model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters, steps_per_epoch)
        for epoch in range(hyperparameters['max_epochs']):
            train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
            val_loss = validate(model, val_loader, criterion, device)
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')
            # Report the validation loss to Optuna
            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss

    except Exception as e:
        print(f"An error occurred: {e}")
        return float('inf')  # Return infinity to indicate a failed trial


def main():
    start_time = time.time()
    optuna.logging.enable_default_handler()
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=10, n_jobs=20)

    end_time = time.time()

    print("Time Elapsed: ", end_time - start_time)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_hyperparams = trial.params
    best_loss = trial.value
    save_results(study, best_hyperparams, best_loss)

    # Plotting
    plot_study_results(study)


def plot_study_results(study):
    """
    Function to plot the final loss for each trial.
    """
    trial_values = [trial.value for trial in study.trials if trial.state == TrialState.COMPLETE]
    plt.figure(figsize=(10, 5))
    plt.plot(trial_values, label='Validation Loss per Trial')
    plt.xlabel('Trial Number')
    plt.ylabel('Val Loss')
    plt.title('Optuna Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparam_results/optuna_trials_loss.png', dpi=300)
    plt.show()

def save_results(study, best_hyperparams, best_loss):
    """
    Save the results of the hyperparameter tuning process along with the best validation loss.
    """
    results = {
        'best_hyperparameters': best_hyperparams,
        'best_loss': best_loss,
        'trials': [
            {
                'hyperparameters': trial.params,
                'value': trial.value,
            } for trial in study.trials if trial.state == TrialState.COMPLETE
        ]
    }
    filename = f'hyperparam_results/optuna_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
