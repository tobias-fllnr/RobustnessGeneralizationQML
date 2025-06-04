import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

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

# Random seeds and regularization parameters
random_ids: list = list(range(50))  # Random IDs from 0 to 49
lambs: list = [round(0.001 * i, 3) for i in range(31)]  # Regularization parameters from 0.0 to 0.03

def load_data(lamb: float, random_id: int, trainable_encoding: bool) -> tuple:
    """
    Loads data for a specific combination of regularization parameter and random seed.

    Args:
        lamb (float): Regularization parameter (lambda).
        random_id (int): Random seed for reproducibility.
        trainable_encoding (bool): Whether the encoding weights are trainable.

    Returns:
        tuple: A tuple containing:
            - lip_bound (float): Lipschitz bound of the model.
            - max_values (np.ndarray): Maximum values of MSE test noise across noise levels.
            - mse_train_loss (float): Final training loss (MSE).
    """
    path: str = f"./Data/Version_{version}/{data_label}/Sequence_length_{seq_length}/Prediction_step_{prediction_step}/Num_qubits_{num_qubits}/Lambda_{lamb}/ID_{random_id}_lr_{learning_rate}_epochs_{epochs}_Trainable_Encoding_{trainable_encoding}"
    loss_metrics: pd.DataFrame = pd.read_csv(path + "/loss_metrics_end.csv")
    lip_bound: float = loss_metrics["Lipschitz Bound"][0]
    mse_test_noise: np.ndarray = np.load(path + "/mse_testing_noise.npy")
    max_values: np.ndarray = np.max(mse_test_noise, axis=1)
    mse_train_loss: float = np.load(path + "/mse_cost_training.npy")[-1]
    return lip_bound, max_values, mse_train_loss

# Initialize lists to store results
lip_bounds: list = []
max_values_list: list = []
mse_train_loss_list: list = []

# Load data for trainable encoding
for lamb in lambs:
    for random_id in random_ids:
        lip_bound, max_values, mse_train_loss = load_data(lamb, random_id, trainable_encoding=True)
        lip_bounds.append(lip_bound)
        max_values_list.append(max_values)
        mse_train_loss_list.append(mse_train_loss)

# Convert lists to numpy arrays and reshape
lip_bounds: np.ndarray = np.array(lip_bounds).reshape(len(lambs), len(random_ids))
max_values_array: np.ndarray = np.array(max_values_list).reshape(len(lambs), len(random_ids), -1)
mse_train_loss_list: np.ndarray = np.array(mse_train_loss_list).reshape(len(lambs), len(random_ids))

# Compute averages and standard deviations
average_lip_bounds: np.ndarray = np.mean(lip_bounds, axis=1)
std_lip_bounds: np.ndarray = np.std(lip_bounds, axis=1)
average_max_values: np.ndarray = np.mean(max_values_array, axis=1)
std_max_values: np.ndarray = np.std(max_values_array, axis=1)
average_mse_train_loss: np.ndarray = np.mean(mse_train_loss_list, axis=1)
std_mse_train_loss: np.ndarray = np.std(mse_train_loss_list, axis=1)

# Load data for fixed encoding (lambda = 0.0)
lip_bounds_fixed: list = []
max_values_list_fixed: list = []
mse_train_loss_list_fixed: list = []

for random_id in random_ids:
    lip_bound, max_values, mse_train_loss = load_data(0.0, random_id, trainable_encoding=False)
    lip_bounds_fixed.append(lip_bound)
    max_values_list_fixed.append(max_values)
    mse_train_loss_list_fixed.append(mse_train_loss)

# Convert lists to numpy arrays
lip_bounds_fixed: np.ndarray = np.array(lip_bounds_fixed)
max_values_array_fixed: np.ndarray = np.array(max_values_list_fixed)
mse_train_loss_list_fixed: np.ndarray = np.array(mse_train_loss_list_fixed)

# Compute averages and standard deviations for fixed encoding
average_lip_bounds_fixed: np.ndarray = np.mean(lip_bounds_fixed, axis=0)
std_lip_bounds_fixed: np.ndarray = np.std(lip_bounds_fixed, axis=0)
average_max_values_fixed: np.ndarray = np.mean(max_values_array_fixed, axis=0)
std_max_values_fixed: np.ndarray = np.std(max_values_array_fixed, axis=0)
average_mse_train_loss_fixed: np.ndarray = np.mean(mse_train_loss_list_fixed, axis=0)
std_mse_train_loss_fixed: np.ndarray = np.std(mse_train_loss_list_fixed, axis=0)

# Plot results
def scientific_formatter(x: float, pos: int) -> str:
    """
    Formats numbers in scientific notation for axis labels.

    Args:
        x (float): Value to format.
        pos (int): Position on the axis.

    Returns:
        str: Formatted string in scientific notation.
    """
    if x == 0:
        return '0'
    exponent: int = int(np.floor(np.log10(abs(x))))
    coeff: float = x / 10**exponent
    return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)

fig, ax1 = plt.subplots(figsize=(5.2, 3.6))

# Plot MSE values on the left Y-axis
ax1.plot(lambs, average_max_values[:, 0], 'D-', color="#1c5d99", label='MSE test', alpha=0.8)
ax1.fill_between(lambs, average_max_values[:, 0] - std_max_values[:, 0] / 2, average_max_values[:, 0] + std_max_values[:, 0] / 2, alpha=0.2, color="#1c5d99")

ax1.plot(lambs, average_mse_train_loss, 'v-', color="#3a77b2", label='MSE train', alpha=0.8)
ax1.fill_between(lambs, average_mse_train_loss - std_mse_train_loss / 2, average_mse_train_loss + std_mse_train_loss / 2, alpha=0.2, color="#3a77b2")

ax1.plot(lambs, average_max_values[:, 0] - average_mse_train_loss, 'h-', color="#6aaee0", label='Generalization gap', alpha=0.8)
ax1.fill_between(lambs, average_max_values[:, 0] - average_mse_train_loss - np.sqrt(std_max_values[:, 0]**2 + std_mse_train_loss**2) / 2, average_max_values[:, 0] - average_mse_train_loss + np.sqrt(std_max_values[:, 0]**2 + std_mse_train_loss**2) / 2, alpha=0.2, color="#6aaee0")

ax1.set_ylabel('MSE', color="C0")
ax1.tick_params(axis='y', labelcolor="C0")
ax1.grid(axis='x', linestyle='--', alpha=0.7)
ax1.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

# Plot Lipschitz Bound on the right Y-axis
ax2 = ax1.twinx()
ax2.plot(lambs, average_lip_bounds, '*-', color="C2", label='Lipschitz Bound', alpha=0.8)
ax2.fill_between(lambs, average_lip_bounds - std_lip_bounds / 2, average_lip_bounds + std_lip_bounds / 2, alpha=0.2, color="C2")
ax2.set_ylabel('Lipschitz Bound', color="C2")
ax2.tick_params(axis='y', labelcolor="C2")

# Common X-axis
ax1.set_xlabel('Regularization parameter $\\lambda$')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=(3,3), fontsize=9)
plt.tight_layout()
plt.savefig(f'./Plots/mse_vs_lambda_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}_generalization_gap.pdf')
