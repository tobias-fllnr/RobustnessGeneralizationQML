import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
lambs: list = [0.0, 0.004, 0.03]  # Regularization parameters

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

# Load data for all lambdas and random IDs
lip_bounds: list = []
max_values_list: list = []
mse_train_loss_list: list = []

for lamb in lambs:
    for random_id in random_ids:
        lip_bound, max_values, mse_train_loss = load_data(lamb, random_id, trainable_encoding=True)
        lip_bounds.append(lip_bound)
        max_values_list.append(max_values)
        mse_train_loss_list.append(mse_train_loss)

# Convert to numpy arrays
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

# Convert to numpy arrays
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

# Main figure and axis
fig, ax = plt.subplots(figsize=(5.5, 4.0))
x_vals: np.ndarray = np.linspace(0, 0.5, 51)

lines: list = []
for i, lamb in enumerate(lambs):
    ax.fill_between(x_vals, average_max_values[i] - std_max_values[i] / 2, average_max_values[i] + std_max_values[i] / 2,
                    color=f"C{i}", alpha=0.3)
    line, = ax.plot(x_vals, average_max_values[i],
                    label=fr'$\lambda = {lamb}$', marker='o', markersize=5, alpha=0.7, color=f"C{i}")
    lines.append(line)

ax.fill_between(x_vals, average_max_values_fixed - std_max_values_fixed / 2, average_max_values_fixed + std_max_values_fixed / 2,
                color=f"C{len(lambs)}", alpha=0.3)

line_fixed, = ax.plot(x_vals, average_max_values_fixed,
                      label='Fixed', marker='o', markersize=5, alpha=0.7, color=f"C{len(lambs)}")
lines.append(line_fixed)

# Configure main plot
ax.set_yscale('log')
ax.set_xlim(0, 0.3)
ax.set_ylim(1e-4, 6e-2)
ax.legend(loc='upper left')
ax.set_xlabel('Noise Level')
ax.set_ylabel('MSE (worst case)')
ax.grid(linestyle='--', alpha=0.7)

# Create inset axes for Lipschitz Bound
inset_ax = inset_axes(ax, width="42%", height="42%", loc='lower right')

x_labels: list = list(range(len(lambs) + 1))
values: list = [x for x in average_lip_bounds] + [average_lip_bounds_fixed]
values = [round(v, 2) for v in values]
colors: list = [line.get_color() for line in lines]

lip_errors: list = [std_lip_bounds[i] for i in range(len(lambs))] + [std_lip_bounds_fixed]
bars = inset_ax.bar(x_labels, values, color=colors, yerr=lip_errors, error_kw={'capsize': 5})
for bar, value in zip(bars, values):
    inset_ax.text(bar.get_x() + bar.get_width() / 2, value * 1.2, f'{value}', ha='center', va='bottom')

# Configure inset plot
inset_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
inset_ax.set_ylim(0, 55)
inset_ax.set_title("Lipschitz Bound", fontsize=10)
inset_ax.tick_params()

# Save the plot
plt.savefig(f"./Plots/mse_vs_noise_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}.pdf", bbox_inches='tight')