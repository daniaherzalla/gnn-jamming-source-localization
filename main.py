import csv
import torch
import pandas as pd
from config import params
from data_processing import load_data, create_data_loader, convert_data_type
from train import initialize_model, train, validate, predict_and_evaluate, predict_and_evaluate_full, plot_network_with_rssi
from utils import set_seeds_and_reproducibility, save_metrics_and_params, save_epochs, save_study_data
import logging
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

# Clear CUDA memory cache
torch.cuda.empty_cache()


def main():
    """
    Main function to run the training and evaluation.
    """
    if params['save_data']:
        seeds = [100]
    else:
        seeds = [1, 42, 23]  # Different seeds for different initialization trials

    for trial_num, seed in enumerate(seeds):
        print("seed: ", seed)
        set_seeds_and_reproducibility(seed)

        # Experiment params
        combination = params['coords'] + '_' + params['edges'] + '_' + params['norm']

        if params['study'] == 'coord_system':
            experiment_path = 'experiments_datasets/cartesian_vs_polar/' + params['dataset'] + params['coords'] + '/'
        elif params['study'] == 'feat_engineering':
            experiment_path = 'experiments_datasets/' + 'engineered_feats/' + params['dataset']
        elif params['study'] == 'dataset':
            experiment_path = 'experiments_datasets/datasets/' + params['dataset']
        else:
            raise "Unknown study type"

        model_path = f'{experiment_path}/trained_model_seed{seed}.pth'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        print("device: ", device)

        train_dataset, val_dataset, test_dataset, original_test_dataset = load_data(params['dataset_path'], params, experiment_path)
        train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])

        # Inference
        if params['inference']:
            steps_per_epoch = len(test_loader)  # Calculate steps per epoch based on the training data loader
            model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch)
            # # Change from str to suitable data type
            # convert_data_type(original_dataset)
            # Load trained model
            model.load_state_dict(torch.load(model_path))
            # Predict jammer position
            predictions, actuals, node_details, err_metrics = predict_and_evaluate_full(test_loader, model, device, original_test_dataset)
            # Save inference data
            study_data = {
                'trial': 'seed_' + str(seed),
                'combination': combination,
                'dataset': params['dataset']
            }
            trial_data = {**study_data, **err_metrics}

            file = f'{experiment_path}/inference_err_metrics.csv'
            save_study_data(trial_data, file)

            # Plot network
            if params['plot_network']:
                for idx, val in enumerate(node_details):
                    plot_network_with_rssi(
                        node_positions=node_details[idx]['node_positions'],
                        final_rssi=node_details[idx]['node_rssi'],
                        jammer_position=node_details[idx]['jammer_position'],
                        noise_floor_db=node_details[idx]['node_noise'],
                        jammed=node_details[idx]['node_states'],
                        prediction=predictions[idx]
                    )
            # return predictions
        else:
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
                'trial': 'seed_' + str(seed),
                'combination': combination,
                'epochs': epoch_info
            }

            # Save the trained model
            print("experiment_path: ", experiment_path)
            torch.save(model.state_dict(), f'{experiment_path}/trained_model_seed{seed}.pth')

            # Evaluate the model on the test set
            model.load_state_dict(best_model_state)
            predictions, actuals, err_metrics, rmse_list = predict_and_evaluate(model, test_loader, device)
            trial_data = {**epoch_data, **err_metrics}
            save_epochs(trial_data, experiment_path)


if __name__ == "__main__":
    main()
