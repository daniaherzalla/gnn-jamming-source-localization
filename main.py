import torch
from config import params
from data_processing import load_data, create_data_loader
from train import initialize_model, train_epoch, validate
from utils import save_metrics_and_params
from utils import set_random_seeds
import logging
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

set_random_seeds()


def main():
    """
    Main function to run the training and validation process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = load_data(params['dataset_path'], params['train_path'], params['val_path'], params['test_path'])
    print("Size of train dataset:", len(train_dataset))
    print("Example data object:", train_dataset[0])
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])
    model, optimizer, scheduler, criterion = initialize_model(device, params)

    best_test_loss = float('inf')
    patience = params['patience']
    epochs_no_improve = 0

    logging.info("Training and validation loop")
    for epoch in range(params['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = validate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping")
                break

        metrics = {
            'best_test_loss': best_test_loss,
            'epochs_trained': epoch
        }

    save_metrics_and_params(metrics, params)

    model.load_state_dict(best_model_state)


if __name__ == "__main__":
    main()
