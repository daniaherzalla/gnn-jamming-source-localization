import json
import torch
from hyperopt import hp, fmin, tpe, Trials, space_eval
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader
from config import params
from utils import set_seeds_and_reproducibility, convert_to_serializable

set_seeds_and_reproducibility()

# Clear CUDA memory cache
# torch.cuda.empty_cache()


def objective(hyperparameters, train_dataset, val_dataset, test_dataset):
    """
    Objective function for hyperparameter optimization.

    Args:
        hyperparameters (dict): Dictionary containing hyperparameters to be optimized.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.

    Returns:
        float: The validation loss after training with the given hyperparameters.
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Create DataLoaders inside the objective function using current hyperparameters for batch size
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, hyperparameters['batch_size'])
    steps_per_epoch = len(train_loader)
    model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters, steps_per_epoch)

    for epoch in range(hyperparameters['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    return val_loss


def map_indices_to_values(hyperparameters):
    """
    Map indices to actual values for hyperparameters that use hp.choice.

    Args:
        hyperparameters (dict): Dictionary of hyperparameters with indices.

    Returns:
        dict: Dictionary of hyperparameters with actual values.
    """
    actual_values = {
        "num_heads": [2, 4, 8],
        "batch_size": [32, 64, 128],
        "max_epochs": [100, 150, 200]
    }
    return {key: actual_values[key][hyperparameters[key][0]] if key in actual_values else hyperparameters[key][0] for key in hyperparameters}


def save_results(trials, best_hyperparams, best_loss, filename='hyperparam_tuning_results.json'):
    """
    Save the results of the hyperparameter tuning process along with the associated best validation loss.

    Args:
        trials (hyperopt.Trials): The trials object containing information about all the trial runs.
        best_hyperparams (dict): The best hyperparameters found during the tuning process.
        best_loss (float): The validation loss associated with the best hyperparameters.
        filename (str): The file path to save the results to.

    Returns:
        None
    """
    results = {
        'best_hyperparameters': convert_to_serializable(best_hyperparams),
        'best_loss': convert_to_serializable(best_loss),
        'trials': [
            {
                'hyperparameters': map_indices_to_values(trial['misc']['vals']),
                'result': convert_to_serializable(trial['result'])
            } for trial in trials.trials
        ]
    }
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    # Load data and preprocess it once
    train_dataset, val_dataset, test_dataset = load_data(params["dataset_path"], params['train_path'], params['val_path'], params['test_path'])

    # Define the hyperparameter space including batch size as a parameter
    hyperparameter_space = {
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'num_heads': hp.choice('num_heads', [2, 4, 8]),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'weight_decay': hp.loguniform('weight_decay', -6, -1),
        'max_epochs': hp.choice('max_epochs', [100, 150, 200])
    }

    trials = Trials()
    best_hyperparameters = fmin(
        fn=lambda hyperparameters: objective(hyperparameters, train_dataset, val_dataset, test_dataset),
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    best_hyperparams = space_eval(hyperparameter_space, best_hyperparameters)
    best_loss = min([trial['result']['loss'] for trial in trials.trials if trial['result']['status'] == 'ok'])

    # Save both results and best hyperparameters along with the best loss
    save_results(trials, best_hyperparams, best_loss)


if __name__ == "__main__":
    main()
