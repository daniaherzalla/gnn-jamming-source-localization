import csv
import os
import pickle
import torch
import statistics
import logging
from config import params
from data_processing import load_data, create_data_loader
from train import initialize_model, train, validate
from utils import set_seeds_and_reproducibility, save_epochs
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

# Clear CUDA memory cache
torch.cuda.empty_cache()


def main():
    """
    Main function to run the training and evaluation.
    """
    seeds = [1, 2, 3]

    if params['train_per_class']:
        if params['dynamic']:
            dataset_classes = ['dynamic_linear_path', 'dynamic_controlled_path']
        else:
            dataset_classes = ['circle_jammer_outside_region','triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
                               'random_jammer_outside_region', 'circle', 'triangle', 'rectangle', 'random']
    else:
        dataset_classes = ["combined"]

    for data_class in dataset_classes:
        rmse_vals = []
        mae_vals = []

        if params['train_per_class']:
            train_set_name = data_class + "_train_dataset.pkl"
            val_set_name = data_class + "_val_dataset.pkl"
            test_set_name = data_class + "_test_dataset.pkl"
        else:
            train_set_name = "train_dataset.pkl"
            val_set_name = "val_dataset.pkl"
            test_set_name = "test_dataset.pkl"

        print("train_set_name: ", train_set_name)
        print("val_set_name: ", val_set_name)
        print("test_set_name: ", test_set_name)

        print("max_nodes: ", params['max_nodes'])
        print("ds_method: ", params['ds_method'])

        for trial_num, seed in enumerate(seeds):
            print("\nseed: ", seed)
            set_seeds_and_reproducibility(100)

            # Experiment params
            combination = params['coords'] + '_' + params['edges'] + str(params['num_neighbors']) + '_' + params['norm'] + '_' + str(params['max_nodes']) + params['ds_method']

            if params['study'] == 'coord_system':
                experiment_path = 'experiments_datasets/cartesian_vs_polar/' + params['experiments_folder']
            elif params['study'] == 'knn_edges':
                experiment_path = 'experiments_datasets/knn_edges/' + params['experiments_folder']
            elif params['study'] == 'feat_engineering':
                experiment_path = 'experiments_datasets/engineered_feats/' + params['experiments_folder']
            elif params['study'] == 'dataset':
                experiment_path = 'experiments_datasets/datasets/' + params['experiments_folder']
            elif params['study'] == 'downsampling':
                experiment_path = 'experiments_datasets/downsampling/' + params['experiments_folder']
            elif params['study'] == 'pooling':
                experiment_path = 'experiments_datasets/pooling/' + params['experiments_folder']
            else:
                raise "Unknown study type"

            # Set path to save or load model
            model_path = f"{experiment_path}trained_model_seed{seed}_{params['model']}_{combination}_{data_class}.pth"

            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device: ", device)

            model_name = params['model']

            # Inference
            if params['inference']:
                for test_set_name in params['test_sets']:
                    print(test_set_name)
                    train_dataset, val_dataset, test_dataset = load_data(params, test_set_name, experiment_path)
                    _, _, test_loader, deg_histogram = create_data_loader(params, train_dataset, val_dataset, test_dataset, experiment_path)
                    steps_per_epoch = len(test_loader)  # Calculate steps per epoch based on the training data loader
                    model, optimizer, scheduler, criterion = initialize_model(device, params, steps_per_epoch, deg_histogram)

                    # Load trained model
                    model.load_state_dict(torch.load(model_path))
                    # Predict jammer position
                    predictions, actuals, err_metrics, perc_completion_list = validate(model, test_loader, criterion, device, test_loader=True)

                    # Save predictions, actuals, and perc_completion to a CSV file
                    max_nodes = params['max_nodes']
                    file = f'{experiment_path}/predictions_{max_nodes}_{test_set_name}_{model_name}'

                    # Check if the file exists
                    file_exists = os.path.isfile(file)
                    with open(file, 'a', newline='') as f:
                        writer = csv.writer(f)

                        # If the file doesn't exist, write the header
                        if not file_exists:
                            writer.writerow(['Seed', 'Prediction', 'Actual', 'Percentage Completion'])

                        # Write the prediction, actual, and percentage completion data
                        for pred, act, perc in zip(predictions, actuals, perc_completion_list):
                            writer.writerow([seed, pred, act, perc])
            else:
                # Load the datasets
                train_dataset, val_dataset, test_dataset = load_data(params, data_class, experiment_path)
                set_seeds_and_reproducibility(seed)

                # Create data loaders
                train_loader, val_loader, test_loader, deg_histogram = create_data_loader(params, train_dataset, val_dataset, test_dataset, experiment_path)

                # Initialize model
                steps_per_epoch = len(train_loader) # Calculate steps per epoch based on the training data loader  # steps per epoch set based on 10000 samples dataset
                model, optimizer, scheduler, regression_criterion, classification_criterion = initialize_model(device, params, steps_per_epoch, deg_histogram)

                best_val_loss = float('inf')

                logging.info("Training and validation loop")
                epoch_info = []
                train_details ={}
                val_details ={}
                for epoch in range(params['max_epochs']):
                    train_loss, train_detailed_metrics = train(model, train_loader, optimizer, regression_criterion, classification_criterion, device, steps_per_epoch, scheduler)
                    val_loss, val_detailed_metrics = validate(model, val_loader, regression_criterion, classification_criterion, device)
                    train_details[epoch] = train_detailed_metrics
                    val_details[epoch] = val_detailed_metrics
                    logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')
                    epoch_info.append(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()

                # Save the trained model
                torch.save(best_model_state, model_path)

                # Save the graph epoch results to a pickle file
                max_nodes = params['max_nodes']
                ds_method = params['ds_method']
                validation_details_path = experiment_path + f'validation_details_{max_nodes}_{ds_method}_seed{seed}.pkl' #_seed{seed}
                train_details_path = experiment_path + f'train_details_{max_nodes}_{ds_method}_seed{seed}.pkl' #_seed{seed}

                with open(validation_details_path, 'wb') as f:
                    pickle.dump(val_details, f)

                with open(train_details_path, 'wb') as f:
                    pickle.dump(train_details, f)

                # Save training conf details
                epoch_data = {
                    'trial': 'seed_' + str(seed),
                    'model': params['model'],
                    'combination': combination,
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'batch_size': params['batch_size'],
                    'dropout_rate': params['dropout_rate'],
                    'num_heads': params['num_heads'],
                    'num_layers': params['num_layers'],
                    'hidden_channels': params['hidden_channels'],
                    'out_channels': params['out_channels'],
                    'train_set': train_set_name,
                    'test_set': test_set_name,
                    'required_feats': params['required_features'],
                    'additional_feats': params['additional_features'],
                    'max_nodes': params['max_nodes'],
                    'ds_method': params['ds_method'],
                    'epochs': epoch_info
                }

                # Evaluate the model on the test set
                model.load_state_dict(best_model_state)
                predictions, actuals, err_metrics, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list = validate(model, test_loader, regression_criterion, classification_criterion, device, test_loader=True)
                trial_data = {**epoch_data, **err_metrics}
                save_epochs(trial_data, experiment_path)
                rmse_vals.append(err_metrics['rmse'])
                mae_vals.append(err_metrics['mae'])  # Collect MAE from err_metrics

                # Save predictions, actuals, and perc_completion to a CSV file
                file = f'{experiment_path}/predictions_{max_nodes}_{ds_method}_{model_name}_{data_class}.csv'

                # Check if the file exists
                file_exists = os.path.isfile(file)
                with open(file, 'a', newline='') as f:
                    writer = csv.writer(f)

                    # If the file doesn't exist, write the header
                    if not file_exists:
                        writer.writerow(['Seed', 'Prediction', 'Actual', 'Percentage Completion', 'PL', 'Sigma', 'JTx', 'Number Samples'])

                    # Write the prediction, actual, and percentage completion data
                    for pred, act, perc, plexp, sigma, jtx, numsamples in zip(predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list):
                        writer.writerow([seed, pred, act, perc, plexp, sigma, jtx, numsamples])

        mean_rmse = statistics.mean(rmse_vals)
        std_rmse = statistics.stdev(rmse_vals)
        print("rmse_vals: ", rmse_vals)
        print(f"Average RMSE: {round(mean_rmse, 1)}\\sd{{{round(std_rmse, 1)}}}\n")

        # MAE statistics calculation
        mean_mae = statistics.mean(mae_vals)
        std_mae = statistics.stdev(mae_vals)
        print("MAE values: ", mae_vals)
        print(f"Average MAE: {round(mean_mae, 1)}\\sd{{{round(std_mae, 1)}}}")

        # Saving RMSE and MAE statistics
        csv_file_path = experiment_path + 'metrics_statistics.csv'
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([params['model'], data_class, f"{round(mean_rmse, 1)}\\sd{{{round(std_rmse, 1)}}}", f"{round(mean_mae, 1)}\\sd{{{round(std_mae, 1)}}}"])

        print(f"Saved RMSE and MAE statistics to {csv_file_path}")


if __name__ == "__main__":
    main()
