import csv
import torch
import pandas as pd
from config import params
from data_processing import load_data, create_data_loader, load_scaler
from train import initialize_model, train, validate, predict_and_evaluate
from utils import set_seeds_and_reproducibility, save_metrics_and_params, save_epochs
import logging
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

set_seeds_and_reproducibility()

# Clear CUDA memory cache
torch.cuda.empty_cache()


def main():
    """
    Main function to run the training and evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("device: ", device)

    train_dataset, val_dataset, test_dataset = load_data(params['dataset_path'], params['train_path'], params['val_path'], params['test_path'])
    # quit()
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])
    steps_per_epoch = len(train_loader)  # Calculate steps per epoch based on the training data loader
    print("steps per epoch: ", steps_per_epoch)
    model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch)

    # # Check model init weights
    # model.print_weights()
    # quit()

    best_val_loss = float('inf')

    logging.info("Training and validation loop")
    epoch_info = []
    for epoch in range(params['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device)  # calculate validation loss to determine if the model is improving during training
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        epoch_info.append(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    combination = params['feats'] + ' ' + params['edges'] + ' ' + params['norm']
    epoch_data = {
        'trial': 'trial_' + str(params['trial_num']),
        'combination': combination,
        'epochs': epoch_info
    }

    metrics = {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch
    }

    save_epochs(epoch_data)
    save_metrics_and_params(metrics, params)

    # # Save the trained model
    # torch.save(model.state_dict(), 'results/trained_model.pth')
    #
    # # Evaluate the model on the test set
    # model.load_state_dict(best_model_state)
    # scaler = load_scaler()
    # predictions, actuals = predict_and_evaluate(model, test_loader, device, scaler)

    # if len(predictions) != len(actuals):
    #     raise ValueError("Predictions and actuals lists must have the same length")

    # # Save predictions and actuals to a CSV file
    # with open('results/predictions.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Prediction', 'Actual'])
    #     for pred, act in zip(predictions, actuals):
    #         writer.writerow([pred, act])  # prediction-actual pair


if __name__ == "__main__":
    main()
