# Jamming Source Localization using GNNs

## Project Overview
This project focuses on localizing a single static jamming source in a swarm of drones using a Graph Neural Network (GNN) to estimate the position. A potential extension of this project can include estimating the jammer type (omnidirectional or directional) and transmit power.

## Model Architecture
The current GNN model employed in this project utilizes Graph Attention Networks (GATs) to dynamically adapt to varying signal patterns within a drone swarm affected by jamming. The model's structure is designed to capture and process the unique characteristics of each node and its connections as follows:

**Input Layer:** Processes node features, which include positional data, received signal strength, jamming status, and the distance from the centroid of the swarm.   
**Edges Definition:** Nodes are interconnected based on geographical proximity. The weight of each edge is determined by the average of their normalized RSSI values.      
**GAT Layers:** Incorporates two layers of GAT.   
**Attention Pooling Layer:** Instead of traditional global mean pooling, the model employs attention pooling to aggregate node features across the graph. This method utilizes the learned attention mechanisms to weigh node features dynamically.   
**Output Layer:** Outputs the estimated jamming source coordinates as (x, y, z), leveraging the graph representation derived from the attention pooling.

## Repository Structure

```
gnn-jamming-source-localization/
│
├── data/                      # Folder containing preprocessed data splits for reproducibility
│   ├── train_dataset.pkl      # Training dataset pickle file
│   ├── test_dataset.pkl       # Test dataset pickle file
│   ├── validation_dataset.pkl # Validation dataset pickle file
│   ├── scalar.pkl             # Scaler object used for normalizing/denormalizing data
├── results/                   # Folder for storing results and metrics
│   ├── model_metrics_and_params.csv  # Metrics and parameters for various model runs
│   ├── hyperparameter_tuning_results.json  # Results of hyperparameter tuning
├── data_processing.py         # Preprocesses the CSV data for training the model.
├── model.py                   # Defines the GNN model for jammer localization.
├── train.py                   # Contains functions to train the GNN model.
├── hyperparam_tuning.py       # Script for hyperparameter optimization using Hyperopt.
├── utils.py                   # Helper functions used across the project.
├── custom_logging.py          # Custom logging utilities for debugging and tracking.
└── config.py                  # Configuration settings for model and training.
```

## Installation

To set up this project, you need to install the required Python libraries. Run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

Note: Ensure you have Python 3.8 or above installed on your system.

## Usage

To preprocess the data and train the model, run:

```bash
python main.py
```

For hyperparameter tuning:

```bash
python hyperparam_tuning.py
```

## Configurations

Adjust the `config.py` file to set various parameters such as the number of epochs, batch size, learning rate, and other model-specific settings. 
The values are currently set based on the outcome of the hyperparameter tuning logged in `hyperparameter_tuning_results.json`.

## Dependencies

- PyTorch
- PyTorch Geometric
- Hyperopt
- Numpy
- Pandas
