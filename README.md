# Graph Attention Network for Jammer Localization in Drone Swarms

## Overview
This project utilizes a Graph Attention Network (GAT) to localize jammers in a simulated drone swarm environment. The GAT model processes spatial coordinates, RSSI values, and other features to predict the jammer's position within the swarm. 

## Model Architecture

The model is structured around a Graph Attention Network (GAT). This architecture is particularly effective for understanding complex spatial relationships and interaction patterns in scenarios where the data inherently forms a graph, such as in drone swarms under jamming attacks. The architecture is detailed as follows:

- **Input Layer**: Each node in the graph represents a data sample from a drone, characterized by features such as spatial coordinates, RSSI values, jamming status, and distance from center of the swarm. These features are first processed through a linear transformation.

- **Graph Attention Layers**: The core of the model consists of multiple GAT layers. Each layer includes:
  - **Attention Mechanisms**: Attention coefficients are calculated using a shared attention mechanism that considers pairs of nodes. This mechanism allows the model to focus on more informative parts of the graph dynamically.
  - **Feature Aggregation**: Node features are updated by aggregating neighbor features weighted by the learned attention coefficients. This aggregation is performed independently for each attention head.
  - **Multi-Head Attention**: The model employs several parallel attention mechanisms (heads) to enhance the stability and capacity of the learning process. The outputs of these heads are concatenated and can be passed through further GAT layers or directed towards the output.

- **Output Layer**: The final set of features, after passing through multiple attention layers, is processed to predict the jammer's coordinates. This output can be customized to also include predictions for the jammer's type and transmit power in future iterations of the project.

- **Regularization and Non-linearities**: LeakyReLU is applied between layers and dropout is applied at the last layer to prevent overfitting and introduce non-linearity.

## Repository Structure

```
gnn-jamming-source-localization/
│
├── data/                       # Folder containing preprocessed data splits for reproducibility
│   ├── static_swarm_3d.csv     # Static swarm dataset used for model training (1000 scenarios)
│   ├── train.gzip              # Training dataset (compressed pickle file)
│   ├── test.gzip               # Test dataset (compressed pickle file)
│   ├── validation.gzip         # Validation dataset (compressed pickle file)
│   ├── scaler.pkl              # Scaler object used for normalizing/denormalizing data
├── results/                    # Folder for storing results and metrics
│   ├── hyperparam_tuning_results.json  # Results of hyperparameter tuning
│   ├── model_metrics_and_params.csv    # Metrics and parameters from various model runs
│   ├── predictions.csv                 # Predicted vs actual jammer positions from test data
├── main.py                     # Orchestrates the training and evaluation processes, including data loading and model initialization
├── train.py                    # Contains the training and validation loops, and functions for model evaluation
├── data_processing.py          # Handles data loading, preprocessing, and transformation into formats suitable for PyTorch Geometric
├── model.py                    # Defines the Graph Attention Network and other neural network architectures used in the project
├── hyperparam_tuning.py        # Manages hyperparameter optimization using Hyperopt.
├── utils.py                    # Provides utility functions 
├── custom_logging.py           # Configures custom logging formats, enhancing output readability during execution
└── config.py                   # Stores configuration parameters that dictate the model's runtime behavior
```

## Installation

To set up this project, you need to install the required Python libraries. Run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

Note: Ensure you have Python 3.8 or above installed on your system.

## Usage

For hyperparameter tuning:

```bash
python hyperparam_tuning.py
```

To initiate the data processing, model training, and final evaluations, run:

```bash
python main.py
```

## Configurations

Adjust the `config.py` file to set various parameters such as the number of epochs, batch size, learning rate, and other model-specific settings. 
The current values are set based on the best hyperparameter results logged in `data/hyperparam_tuning_results.json`.

