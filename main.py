import csv
import torch
from config import params
from data_processing import load_data, create_data_loader, load_scaler
from train import initialize_model, train_epoch, validate, predict_and_evaluate
from utils import set_random_seeds, save_metrics_and_params
import logging
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

set_random_seeds()


def main():
    """
    Main function to run the training and evaluation process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = load_data(params['dataset_path'], params['train_path'], params['val_path'], params['test_path'])
    # print("Size of train dataset:", len(train_dataset))
    # print("Example data object:", train_dataset[0])
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])
    model, optimizer, scheduler, criterion = initialize_model(device, params)

    best_val_loss = float('inf')
    patience = params['patience']
    epochs_no_improve = 0

    logging.info("Training and validation loop")
    for epoch in range(params['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # use the validate function to calculate validation loss and determine if the model is improving during training
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping")
                break

    metrics = {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch
    }

    save_metrics_and_params(metrics, params)

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

    model.load_state_dict(best_model_state)

    # Evaluate the model on the test set
    scaler = load_scaler()  # Adjust the path to your scaler file
    predictions, actuals = predict_and_evaluate(model, test_loader, device, scaler)

    # Ensure both lists have the same length
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals lists must have the same length")

    # Save predictions and actuals to a CSV file
    with open('results/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction', 'Actual'])  # Write the header
        for pred, act in zip(predictions, actuals):
            writer.writerow([pred, act])  # Write each prediction-actual pair


if __name__ == "__main__":
    main()
