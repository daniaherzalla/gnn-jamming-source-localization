import csv
import torch
import statistics
import logging
from config import params
from data_processing import load_data, create_data_loader
from train import initialize_model, train, validate, predict_and_evaluate
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
    seeds = [1, 23, 42]

    if params['train_per_class']:
        dataset_classes = ['circle', 'triangle', 'rectangle', 'random',
        'all_jammed', 'circle_jammer_outside_region',
        'triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
        'random_jammer_outside_region', 'all_jammed_jammer_outside_region']
    else:
        dataset_classes = ["combined"]

    for data_class in dataset_classes:
        rmse_vals = []

        if params['train_per_class']:
            train_set_name = data_class + "_train_set.csv"
            val_set_name = data_class + "_val_set.csv"
            test_set_name = data_class + "_test_set.csv"
        else:
            train_set_name = "train_dataset.csv"
            val_set_name = "val_dataset.csv"
            test_set_name = "test_dataset.csv"

        print("train_set_name: ", train_set_name)
        print("val_set_name: ", val_set_name)
        print("test_set_name: ", test_set_name)

        for trial_num, seed in enumerate(seeds):
            print("\nseed: ", seed)
            set_seeds_and_reproducibility(100)

            # Experiment params
            combination = params['coords'] + '_' + params['edges'] + str(params['num_neighbors']) + '_' + params['norm']

            if params['study'] == 'coord_system':
                experiment_path = 'experiments_datasets/cartesian_vs_polar/' + params['experiments_folder']
            elif params['study'] == 'knn_edges':
                experiment_path = 'experiments_datasets/knn_edges/' + params['experiments_folder']
            elif params['study'] == 'feat_engineering':
                experiment_path = 'experiments_datasets/engineered_feats/' + params['experiments_folder']
            elif params['study'] == 'dataset':
                experiment_path = 'experiments_datasets/datasets/' + params['experiments_folder']
            else:
                raise "Unknown study type"

            # Set path to save or load model
            model_path = f"{experiment_path}trained_model_seed{seed}_{params['model']}_{combination}_{data_class}.pth"

            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device: ", device)

            # Load the datasets
            train_dataset, val_dataset, test_dataset = load_data(params, test_set_name, experiment_path)
            set_seeds_and_reproducibility(seed)

            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=params['batch_size'])

            # Initialize model
            steps_per_epoch = len(train_loader)  # Calculate steps per epoch based on the training data loader  # steps per epoch set based on 10000 samples dataset
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
            torch.save(best_model_state, f"{experiment_path}/trained_model_seed{seed}_{params['model']}_{combination}_{data_class}.pth")

            epoch_data = {
                'trial': 'seed_' + str(seed),
                'model': params['model'],
                'combination': combination,
                'window_size': params['window_size'],
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
                'epochs': epoch_info
            }

            # Evaluate the model on the test set
            model.load_state_dict(best_model_state)
            predictions, actuals, err_metrics, perc_completion_list = predict_and_evaluate(model, test_loader, device)
            trial_data = {**epoch_data, **err_metrics}
            save_epochs(trial_data, experiment_path)
            rmse_vals.append(err_metrics['rmse'])

            # Save predictions, actuals, and perc_completion to a CSV file
            file = f'{experiment_path}/predictions.csv'
            with open(file, 'a', newline='') as f:
                writer = csv.writer(f)
                # Add 'Seed' and 'Percentage Completion' to the header row
                writer.writerow(['Seed', 'Prediction', 'Actual', 'Percentage Completion'])
                for pred, act, perc in zip(predictions, actuals, perc_completion_list):
                    # Include the seed and perc_completion in each row
                    writer.writerow([seed, pred, act, perc])


        mean_rmse = statistics.mean(rmse_vals)
        std_rmse = statistics.stdev(rmse_vals)
        print("rmse_vals: ", rmse_vals)
        print(f"Average RMSE: {round(mean_rmse, 1)}\\sd{{{round(std_rmse,1)}}}\n")

        formatted_value = f"{round(mean_rmse, 1)}\\sd{{{round(std_rmse,1)}}}"
        csv_file_path = experiment_path + 'rmse_statistics.csv'

        # Open the CSV file in append mode and write the formatted value
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([params['model'], data_class, formatted_value])


        print(f"Saved: {formatted_value} to {csv_file_path}")



if __name__ == "__main__":
    main()
