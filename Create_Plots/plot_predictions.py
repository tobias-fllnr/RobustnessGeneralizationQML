import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to the system path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.analyzer import Analyzer
from utils.handling_data import DataHandling
from utils.circuit import Circuit
from utils.trainer import Trainer

# Update matplotlib settings for LaTeX rendering and font customization
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Use default LaTeX font
    }
)

# Experiment parameters
version: int = 1
learning_rate: float = 0.01
data_label: str = "logistic_map_1000_chaos"
seq_length: int = 12
prediction_step: int = 1
num_qubits: int = 4
epochs: int = 2000

# Color settings for plots
color_train: str = "C0"  # Blue (colorblind-friendly)
color_test: str = "C2"   # Orange (colorblind-friendly)
color_baseline: str = "#666666"  # Neutral gray for baseline

def plot_predictions(version: int, learning_rate: float, data_label: str, seq_length: int, prediction_step: int, num_qubits: int, epochs: int) -> None:
    """
    Plots the scatter of true vs predicted values for different model configurations.

    Args:
        version (int): Version identifier for the experiment.
        learning_rate (float): Learning rate for the optimizer.
        data_label (str): Label describing the type of data being processed.
        seq_length (int): Sequence length for the input data.
        prediction_step (int): Step into the future for prediction.
        num_qubits (int): Number of qubits in the quantum circuit.
        epochs (int): Number of training epochs.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.1))

    # Model settings for different configurations
    settings: dict = {
        r"$\lambda = 0.0$ trainable enc.": (True, 9, 0.0),
        r"$\lambda = 0.004$ trainable enc.": (True, 30, 0.004),
        r"fixed enc.": (False, 34, 0.0)
    }

    for i, (title, (trainable_encoding, random_id, lamb)) in enumerate(settings.items()):
        ax = axes[i]

        # Initialize model, data handler, trainer, and analyzer
        model: Circuit = Circuit(num_qubits=num_qubits, seq_length=seq_length, data_label=data_label, random_id=random_id, trainable_encoding=trainable_encoding)
        data_handler: DataHandling = DataHandling(data_label=data_label, seq_length=seq_length, prediction_step=prediction_step, random_id=random_id)
        trainer: Trainer = Trainer(model=model, lamb=lamb, learning_rate=learning_rate, epochs=epochs)
        analyzer: Analyzer = Analyzer(version=version, model=model, data_handler=data_handler, trainer=trainer)

        # Load trained model
        model.load_state_dict(torch.load(analyzer.path + "/trained_model"))
        model.eval()

        # Get training and testing data
        inputs_train, labels_train, inputs_test, labels_test = data_handler.get_training_and_test_data()

        # Rescale predictions and labels
        preds_train: np.ndarray = 0.5 * model(inputs_train).detach().numpy() + 3.5
        preds_test: np.ndarray = 0.5 * model(inputs_test).detach().numpy() + 3.5
        labels_train: np.ndarray = 0.5 * labels_train.numpy() + 3.5
        labels_test: np.ndarray = 0.5 * labels_test.numpy() + 3.5

        # Plot true vs predicted values
        ax.scatter(labels_test, preds_test, alpha=0.5, label='Test', marker='.', color=color_test, s=15)
        ax.scatter(labels_train, preds_train, alpha=0.5, label='Train', marker='.', color=color_train, s=15)

        # Plot baseline (perfect prediction line)
        ax.plot([3.5, 4], [3.5, 4], linestyle='--', color=color_baseline, label='Baseline')

        # Configure subplot
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("True Value")
        if i == 0:
            ax.set_ylabel("Predicted Value")
        ax.set_xlim(3.48, 4.02)
        ax.set_ylim(3.48, 4.02)
        ax.legend(fontsize=7, loc='lower right')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"./Plots/prediction_scatter_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}.pdf", bbox_inches='tight')

# Call the function to generate the plot
plot_predictions(version=version, learning_rate=learning_rate, data_label=data_label, seq_length=seq_length, prediction_step=prediction_step, num_qubits=num_qubits, epochs=epochs)