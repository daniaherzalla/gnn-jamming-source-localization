import json
import time

import torch
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader


# TIMEOUT = 52800  # seconds
# TIMEOUT = 48600  # seconds
# TIMEOUT = 50400  # seconds # 14 hours
TIMEOUT = 43200  # seconds # 12 hours
NUM_JOBS = 1


def objective(trial, model):
    """
    Objective function for hyperparameter optimization.
    Attempts to run the trial and returns infinity if an error occurs,
    signaling a failed trial.
    """
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        error_message = f"Error setting device: {e}"
        print(error_message)
        raise SystemExit(error_message)

    print("device: ", device)
    hyperparameters = {
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.7),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'num_layers': trial.suggest_categorical('num_layers', [2, 4, 8]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512]),
        'out_channels': trial.suggest_categorical('out_channels', [32, 64, 128, 256, 512]),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.001),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'num_neighbors': trial.suggest_categorical('num_neighbors', [5, 10, 20, 30]),
        'required_features': ['node_positions', 'node_noise'],
        'max_epochs': 200,
        'coords': 'cartesian',
        'edges': 'knn',
        'norm': 'minmax',
        'inference': False,
        'save_data': False,
        'model': model,
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
        # '/home/mladmin/dania/gnn-jamming-source-localization/data/random.csv'
        train_dataset, val_dataset, test_dataset, original_dataset = load_data('data/combined_fspl_log_distance.csv', hyperparameters)
        train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=hyperparameters['batch_size'])
        steps_per_epoch = len(train_loader)
        model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters, steps_per_epoch)
        for epoch in range(hyperparameters['max_epochs']):
            train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
            val_loss = validate(model, val_loader, criterion, device)
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')
            # Report the validation loss to Optuna
            trial.report(val_loss, epoch)

            # # Report the validation loss only at epoch 75
            # if epoch == 75:
            #     trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value at epoch 75
            if trial.should_prune():
                trial.set_user_attr('pruned_epoch', epoch)
                raise optuna.exceptions.TrialPruned()

        return val_loss

    except optuna.exceptions.TrialPruned as e:
        print(e)
        raise e

    except Exception as e:
        print(f"An error occurred: {e}")
        trial.set_user_attr('error_message', str(e))
        return float('inf')  # Indicate a failed trial


def load_initial_hyperparameters():
    with open('initial_hyperparameters.json', 'r') as file:
        return json.load(file)


def main():
    initial_hyperparameters = load_initial_hyperparameters()
    for model in initial_hyperparameters.keys():
        print(f"Starting hyperparameter tuning for model: {model}")
        optuna.logging.enable_default_handler()
        sampler = optuna.samplers.TPESampler()
        # pruner = optuna.pruners.MedianPruner()
        # Instantiate the HyperbandPruner
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,  # Minimum number of training iterations
            max_resource=200,  # Maximum number of training iterations (as defined by 'max_epochs')
            reduction_factor=3  # Determines how aggressively trials are pruned
        )
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        study.enqueue_trial(initial_hyperparameters[model])
        try:
            start_time = time.time()
            study.optimize(lambda trial: objective(trial, model), timeout=TIMEOUT, n_jobs=NUM_JOBS)  # Adjust the timeout as necessary
            elapsed_time = time.time() - start_time
            print(f"Optimization for model {model} completed in {elapsed_time} seconds")
        except Exception as e:
            print(f"An error occurred during optimization for model {model}: {e}")

        best_hyperparams = study.best_trial.params
        print(f"Best parameters for {model}: {best_hyperparams}")

        best_loss = study.best_trial.value
        save_results(study, model)

        plot_study_results(study, model)


def plot_study_results(study, model):
    """
    Function to plot the validation loss for each epoch of each trial, including pruned trials.
    """
    plt.figure(figsize=(15, 7))  # Set a larger figure size for better visibility
    for trial in study.trials:
        if trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
            # Fetch intermediate values from the trial
            if trial.intermediate_values:
                epochs = list(trial.intermediate_values.keys())
                values = list(trial.intermediate_values.values())

                # For pruned trials, truncate the plot at the pruned epoch
                if trial.state == optuna.trial.TrialState.PRUNED:
                    print(int(trial.user_attrs.get('pruned_epoch', 200)))
                    pruned_epoch = int(trial.user_attrs.get('pruned_epoch', 200))
                    epochs = [e for e in epochs if e <= pruned_epoch]
                    values = values[:len(epochs)]

                plt.plot(epochs, values, label=f'Trial {trial.number} ({trial.state})')

    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Loss')
    plt.title(f'Optuna Optimization Progress for {model}')
    plt.legend(loc='upper right', fontsize=8)  # Adjust legend location and size for better visibility
    plt.grid(True)
    plt.savefig(f'hyperparam_results/latest_optuna/{model}_optuna_epoch_loss.png', dpi=300)


def save_results(study, model):
    results = {
        'model': model,
        'best_hyperparameters': study.best_trial.params,
        'best_loss': study.best_trial.value,
        'trials': []
    }

    for trial in study.trials:
        trial_info = {
            'hyperparameters': trial.params,
            'final_value': trial.value,
            'state': str(trial.state),
            'intermediate_values': trial.intermediate_values
        }

        trial_info['last_epoch'] = trial.user_attrs.get('pruned_epoch', 200)
        results['trials'].append(trial_info)

    with open(f'hyperparam_results/latest_optuna/{model}_optuna_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
