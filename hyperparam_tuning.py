import json
import torch
from hyperopt import hp, fmin, tpe, Trials, space_eval
from train import initialize_model, train_epoch, validate
from data_processing import load_data, create_data_loader
from config import params
from utils import set_random_seeds, convert_to_serializable

set_random_seeds()


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoaders inside the objective function using current hyperparameters for batch size
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, hyperparameters['batch_size'])

    model, optimizer, scheduler, criterion = initialize_model(device, hyperparameters)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(hyperparameters['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparameters['patience']:
                print("Early stopping")
                break

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    return val_loss


def save_results(trials, best_hyperparams, filename='results/hyperparameter_tuning_results.json'):
    """
    Save the results of the hyperparameter tuning process.

    Args:
        trials (hyperopt.Trials): The trials object containing information about all the trial runs.
        best_hyperparams (dict): The best hyperparameters found during the tuning process.
        filename (str): The file path to save the results to.

    Returns:
        None
    """
    # Trials doc: https://github.com/hyperopt/hyperopt/blob/master/hyperopt/base.py
    results = {
        'best_hyperparameters': best_hyperparams,
        'trials': [{'hyperparameters': convert_to_serializable(trial['misc']['vals']), 'result': convert_to_serializable(trial['result'])} for trial in trials.trials]
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
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'weight_decay': hp.loguniform('weight_decay', -6, -1),
        'patience': hp.choice('patience', [10, 20, 30]),
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

    # Save both results and best hyperparameters
    save_results(trials, best_hyperparams)


if __name__ == "__main__":
    main()
