import csv
import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
# torch.use_deterministic_algorithms(True)
import pandas as pd
import statistics
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
        seeds = [1, 23, 42]  # Different seeds for different initialization trials
        # seeds = [1, 7, 23, 42, 12345]
        rmse_vals = []

    for trial_num, seed in enumerate(seeds):
        print("\nseed: ", seed)
        set_seeds_and_reproducibility(seed)

        # Experiment params
        combination = params['coords'] + '_' + params['edges'] + str(params['num_neighbors']) + '_' + params['norm']

        if params['study'] == 'coord_system':
            experiment_path = 'experiments_datasets/cartesian_vs_polar/' + params['experiments_folder'] # + params['coords'] + '/'
        elif params['study'] == 'knn_edges':
            experiment_path = 'experiments_datasets/knn_edges/' + params['experiments_folder']
        elif params['study'] == 'feat_engineering':
            experiment_path = 'experiments_datasets/engineered_feats/' + params['experiments_folder']
        elif params['study'] == 'dataset':
            experiment_path = 'experiments_datasets/datasets/' + params['experiments_folder']
        else:
            raise "Unknown study type"

        model_path = f'{experiment_path}/trained_model_seed{seed}.pth'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        print("device: ", device)

        # Inference
        if params['inference']:
            for test_set in params['test_sets']:
                print(test_set)
                train_dataset, val_dataset, test_dataset, original_test_dataset = load_data(params['dataset_path'], params, test_set, experiment_path)
                train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])

                steps_per_epoch = len(test_loader)  # Calculate steps per epoch based on the training data loader
                model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch)
                # TODO: check how to pass original dataset for plotting - should be csv test set generated
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
                    'dataset': test_set
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
            train_dataset, val_dataset, test_dataset, original_test_dataset = load_data(params['dataset_path'], params, 'test_dataset.pt', experiment_path)
            train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])

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

            # Save the trained model
            print("experiment_path: ", experiment_path)
            if params['additional_features']:
                additional_features = '_' + '_'.join(params['additional_features'])
            else:
                additional_features = ''
            torch.save(best_model_state, f'{experiment_path}/trained_model_seed{seed}_{combination}{additional_features}.pth')

            epoch_data = {
                'trial': 'seed_' + str(seed),
                'combination': combination,
                'train_set': params['train_set'],
                'test_set': params['test_set'],
                'additional_feats': params['additional_features'],
                'epochs': epoch_info
            }

            # Evaluate the model on the test set
            model.load_state_dict(best_model_state)
            predictions, actuals, err_metrics, rmse_list = predict_and_evaluate(model, test_loader, device)
            trial_data = {**epoch_data, **err_metrics}
            save_epochs(trial_data, experiment_path)
            rmse_vals.append(err_metrics['rmse'])

    mean_rmse = statistics.mean(rmse_vals)
    std_rmse = statistics.stdev(rmse_vals)
    print("rmse_vals: ", rmse_vals)
    print(f"Average RMSE: {round(mean_rmse, 1)}\\sd{{{round(std_rmse,1)}}}\n")


if __name__ == "__main__":
    main()
