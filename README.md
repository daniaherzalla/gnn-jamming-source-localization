# Graph Neural Networks for Jamming Source Localization

This project addresses the challenge of jamming source localization in wireless networks by leveraging graph-based learning. Traditional geometric optimization techniques struggle under environmental uncertainties and dense interference. Our approach integrates structured node representations and an attention-based GNN to ensure spatial coherence and adaptive signal fusion. The framework is evaluated under complex RF environments, demonstrating superior performance compared to established baselines.

### Key Features

- **Graph-Based Learning**: Reformulates jamming source localization as an inductive graph regression task.
- **Attention-Based GNN**: Utilizes an attention mechanism to refine neighborhood influence and improve robustness.
- **Confidence-Guided Estimation**: Dynamically balances GNN predictions with domain-informed priors for adaptive localization.


## Global Context Encoding
The graph representation is extended by introducing a supernode that encodes a structured global prior based on noise floor levels. This supernode acts as a global aggregator, influencing the computation of confidence weights while remaining decoupled from the GNN regression process.

## Confidence-Guided Adaptive Position Estimation

The final jammer position is estimated as a weighted combination of the GNN-based prediction and a domain-informed Weighted Centroid Localization (WCL) prior. This adaptive mechanism ensures robustness across varying sampling densities and spatial distributions. 
The confidence weights $\alpha$ are computed as:

$$
\alpha = \sigma(W_\alpha h_{\text{super}} + b_\alpha)
$$

where $ \alpha \in \mathbb{R}^5 $ is a five-dimensional confidence vector. The final predicted jammer position is:

$$
\hat{x}_{\text{final}} = \alpha \odot \hat{x}_{\text{GNN}} + (1 - \alpha) \odot \hat{x}_{\text{WCL}}
$$


## Training Strategy

The training process minimizes a joint loss function that balances the GNN-based estimate and the WCL prior. The adaptive estimation loss is defined as:

$$
\mathcal{L}_{\text{Adapt}} = \frac{1}{|B|} \sum_{m \in B} \left\| \hat{x}^{(m)}_j - \left( \alpha^{(m)} \odot \hat{x}^{(m)}_{\text{GNN}} + (1 - \alpha^{(m)}) \odot \hat{x}^{(m)}_{\text{WCL}} \right) \right\|^2
$$

where $ \hat{x}^{(m)}_j $ is the ground truth jammer position. To ensure the GNN independently learns to predict the jammer's position, an additional loss term is introduced:

$$
\mathcal{L}_{\text{GNN}} = \frac{1}{|B|} \sum_{m \in B} \left\| \hat{x}^{(m)}_j - \hat{x}^{(m)}_{\text{GNN}} \right\|^2
$$

The joint loss function is:

$$
\mathcal{L}_{\text{CAGE}} = \frac{1}{2} (\mathcal{L}_{\text{GNN}} + \mathcal{L}_{\text{Adapt}}) + \lambda \sum_{m \in B} (1 - \alpha^{(m)})^2
$$

where $\lambda$ is a hyperparameter controlling the penalty for over-reliance on the WCL prior.


## Project Structure

The project is organized as follows:

- **`custom_logging.py`**: Custom logging setup for tracking and debugging.
- **`global_config.py`**: Global configuration and argument parsing for the project.
- **`main.py`**: Entry point for running the project. Supports training, validation, and inference.
- **`data_processing.py`**: Handles data loading, preprocessing, and DataLoader creation.
- **`model.py`**: Defines the GNN architecture and related components.
- **`train.py`**: Implements the training and validation loops.
- **`utils.py`**: Utility functions for reproducibility, saving results, and other helper tasks.
- **`data/dynamic_data.pkl`**: Dynamic jammed network containing raw position and noise floor data.
- **`experiments/`**: Directory to store experiment results, trained models, and logs.

## Running the Project

The project supports various command-line arguments for configuring model training, data preprocessing, and experiments. To view all available options, run:
```bash
python main.py --help
```

To run the project with default CAGE parameters, run:

```bash
python main.py
```


