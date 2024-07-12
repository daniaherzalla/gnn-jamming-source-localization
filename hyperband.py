import json
import torch
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
from ray.tune.schedulers import HyperBandScheduler
from train import initialize_model, train, validate
from data_processing import load_data, create_data_loader
from config import params
import random
import ray

# ray.init(num_gpus=1)


def sample_additional_features(config):
    """
    Randomly sample additional features with a given probability.
    Accepts a config dictionary but does not use it, allowing integration with Ray Tune.
    """
    features = ['dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise']  # 'node_states',
    if random.random() < 0.8:
        num_features = random.randint(1, len(features))
        return random.sample(features, num_features)
    return []


def objective(config):
    """
    Objective function adapted for Ray Tune, now including 'additional_features'.
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check if Ray has allocated a GPU
    device = torch.device('cuda' if ray.get_gpu_ids() else 'cpu')
    print('device:', device)

    train_dataset, val_dataset, test_dataset, _ = load_data(params['dataset_path'], config)
    train_loader, val_loader, test_loader = create_data_loader(train_dataset, val_dataset, test_dataset, batch_size=config['batch_size'])
    steps_per_epoch = len(train_loader)
    model, optimizer, scheduler, criterion = initialize_model(device, config, steps_per_epoch)

    for epoch in range(params['max_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, steps_per_epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device)
        session.report({"loss": val_loss, "config": config})


def main(use_tpe=True):
    hyperparameter_space = {
        'dropout_rate': tune.uniform(0.1, 0.6),
        'num_heads': tune.choice([2, 4, 8]),
        'num_layers': tune.choice([2, 4, 8]),
        'hidden_channels': tune.choice([32, 64, 128, 256]),
        'out_channels': tune.choice([32, 64, 128, 256]),
        'batch_size': tune.choice([16, 32, 64]),
        'learning_rate': tune.loguniform(1e-4, 1e-2),
        'weight_decay': tune.loguniform(1e-6, 1e-2),
        'num_neighbors': tune.choice([5, 10, 20, 30]),
        'required_features': ['node_positions', 'node_noise'],
        'additional_features': tune.sample_from(sample_additional_features),
        'max_epochs': tune.choice([200]),
        'coords': tune.choice(['cartesian']),
        'edges': tune.choice(['knn']),
        'norm': tune.choice(['minmax']),
        'model': tune.choice(['GATv2'])
    }

    if use_tpe:
        tpe_sampler = TPESampler()  # Explicitly define the TPE sampler
        search_alg = OptunaSearch(sampler=tpe_sampler, metric="loss", mode="min")
        scheduler = None
    else:
        search_alg = None
        scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            max_t=150,
            reduction_factor=3
        )

    analysis = tune.run(
        objective,
        config=hyperparameter_space,
        num_samples=5,
        search_alg=search_alg,
        scheduler=scheduler,
        metric="loss",
        mode="min",
        resources_per_trial={"gpu": 1, "cpu": 1}
    )

    # Extract trial data
    trials_data = []
    for trial in analysis.trials:
        trial_data = {
            "config": trial.config,
            "loss": trial.last_result["loss"],
            "status": trial.status,
            "local_path": trial.local_path
        }
        trials_data.append(trial_data)

    # Optional: Process and save the best trial data
    best_trial = analysis.get_best_trial("loss", "min", "last")
    model_name = best_trial.config['model']

    # Save all trial data to a JSON file
    all_trials_filename = f"hyperparam_results/all_trial_results_{model_name}.json"
    with open(all_trials_filename, "w") as f:
        json.dump(trials_data, f, indent=4)

    results = {
        "best_config": best_trial.config,
        "best_loss": best_trial.last_result['loss']
    }
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trial_filename = f"hyperparam_results/best_trial_results_{model_name}.json"
    with open(best_trial_filename, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Set use_tpe=True to use TPE, or False to use HyperBand
    main(use_tpe=True)
