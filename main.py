import csv
import torch
import pandas as pd
from config import params
from data_processing import load_data, create_data_loader, convert_data_type
from train import initialize_model, train, validate, predict_and_evaluate, predict_and_evaluate_full, plot_network_with_rssi
from utils import set_seeds_and_reproducibility, save_metrics_and_params, save_epochs
import logging
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

if params['reproduce']:
    set_seeds_and_reproducibility()

# Clear CUDA memory cache
torch.cuda.empty_cache()


def main():
    """
    Main function to run the training and evaluation.
    """
    # Experiment params
    combination = params['coords'] + '_' + params['edges'] + '_' + params['norm']
    experiment_path = 'experiments/' + combination + '/trial' + str(params['trial_num'])
    model_path = f'{experiment_path}/trained_model_{combination}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("device: ", device)

    train_dataset, val_dataset, test_dataset, original_dataset = load_data(params['dataset_path'], params)
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])

    # Inference
    if params['inference']:
        steps_per_epoch = len(test_loader)  # Calculate steps per epoch based on the training data loader
        model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch)
        # Change from str to suitable data type
        convert_data_type(original_dataset)
        # Load trained model
        model.load_state_dict(torch.load(model_path))
        # Predict jammer position
        predictions, actuals, node_details = predict_and_evaluate_full(test_loader, model, device, original_dataset)
        # Plot network
        for idx, val in enumerate(node_details):
            plot_network_with_rssi(
                node_positions=node_details[idx]['node_positions'],
                final_rssi=node_details[idx]['node_rssi'],
                jammer_position=node_details[idx]['jammer_position'],
                noise_floor_db=node_details[idx]['node_noise'],
                jammed=node_details[idx]['node_states'],
                prediction=predictions[idx]
            )
        return predictions

    # Initialize model
    steps_per_epoch = len(train_loader)  # Calculate steps per epoch based on the training data loader
    model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch)

    best_val_loss = float('inf')

    logging.info("Training and validation loop")
    epoch_info = []
    for epoch in range(params['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device)  # calculate validation loss
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')
        epoch_info.append(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    epoch_data = {
        'trial': 'trial_' + str(params['trial_num']),
        'combination': combination,
        'epochs': epoch_info
    }

    metrics = {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch
    }
    save_metrics_and_params(metrics, params)

    # Save the trained model
    torch.save(model.state_dict(), f'{experiment_path}/trained_model_{combination}.pth')

    # Evaluate the model on the test set
    model.load_state_dict(best_model_state)
    predictions, actuals, err_metrics, rmse_list = predict_and_evaluate(model, test_loader, device)
    trial_data = {**epoch_data, **err_metrics}
    save_epochs(trial_data)

    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals lists must have the same length")

    # Save predictions and actuals to a CSV file
    with open(f'{experiment_path}/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction', 'Actual', 'RMSE'])
        for pred, act, rmse in zip(predictions, actuals, rmse_list):
            writer.writerow([pred, act, rmse])  # prediction-actual pair


if __name__ == "__main__":
    main()
