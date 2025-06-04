import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Use default LaTeX font
    }
)

version = 1
learning_rate = 0.01
data_label = "logistic_map_1000_chaos"
seq_length = 12
prediction_step = 1
num_qubits = 4
epochs = 2000

random_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
lambs = [0.0, 0.004, 0.03]

def load_data(lamb, random_id, trainable_encoding):
    path =f"./Data/Version_{version}/{data_label}/Sequence_length_{seq_length}/Prediction_step_{prediction_step}/Num_qubits_{num_qubits}/Lambda_{lamb}/ID_{random_id}_lr_{learning_rate}_epochs_{epochs}_Trainable_Encoding_{trainable_encoding}"
    loss_metrics = pd.read_csv(path + "/loss_metrics_end.csv")
    lip_bound = loss_metrics["Lipschitz Bound"][0]
    mse_test_noise = np.load(path + "/mse_testing_noise.npy")
    max_values = np.max(mse_test_noise, axis=1)
    mse_train_loss = np.load(path + "/mse_cost_training.npy")
    mse_train_loss = mse_train_loss[-1]
    return lip_bound, max_values, mse_train_loss

# Load data for all lambdas and random IDs
lip_bounds = []
max_values_list = []
mse_train_loss_list = []

for lamb in lambs:
    for random_id in random_ids:
        lip_bound, max_values, mse_train_loss = load_data(lamb, random_id, trainable_encoding=True)
        lip_bounds.append(lip_bound)
        max_values_list.append(max_values)
        mse_train_loss_list.append(mse_train_loss)

# Convert to numpy arrays
lip_bounds = np.array(lip_bounds)  # shape: (len(lambs) * len(random_ids),)
max_values_array = np.array(max_values_list)  # shape: (len(lambs) * len(random_ids), n)
mse_train_loss_list = np.array(mse_train_loss_list)  # shape: (len(lambs) * len(random_ids),)

lip_bounds = lip_bounds.reshape(len(lambs), len(random_ids))
max_values_array = max_values_array.reshape(len(lambs), len(random_ids), -1)
mse_train_loss_list = mse_train_loss_list.reshape(len(lambs), len(random_ids))



average_lip_bounds = np.mean(lip_bounds, axis=1)
std_lip_bounds = np.std(lip_bounds, axis=1)
average_max_values = np.mean(max_values_array, axis=1)
std_max_values = np.std(max_values_array, axis=1)
average_mse_train_loss = np.mean(mse_train_loss_list, axis=1)
std_mse_train_loss = np.std(mse_train_loss_list, axis=1)


# Load Data for fixed lamb=0.0 case
lip_bounds_fixed = []
max_values_list_fixed = []
mse_train_loss_list_fixed = []

for random_id in random_ids:
    lip_bound, max_values, mse_train_loss = load_data(0.0, random_id, trainable_encoding=False)
    lip_bounds_fixed.append(lip_bound)
    max_values_list_fixed.append(max_values)
    mse_train_loss_list_fixed.append(mse_train_loss)

# Convert to numpy arrays
lip_bounds_fixed = np.array(lip_bounds_fixed) 
max_values_array_fixed = np.array(max_values_list_fixed)  
mse_train_loss_list_fixed = np.array(mse_train_loss_list_fixed)

average_lip_bounds_fixed = np.mean(lip_bounds_fixed, axis=0)
std_lip_bounds_fixed = np.std(lip_bounds_fixed, axis=0)

average_max_values_fixed = np.mean(max_values_array_fixed, axis=0)
std_max_values_fixed = np.std(max_values_array_fixed, axis=0)

average_mse_train_loss_fixed = np.mean(mse_train_loss_list_fixed, axis=0)
std_mse_train_loss_fixed = np.std(mse_train_loss_list_fixed, axis=0)


# Main figure and axis
fig, ax = plt.subplots(figsize=(5.5, 4.0))
x_vals = np.linspace(0, 0.5, 51)

lines = []
for i, lamb in enumerate(lambs):
    ax.fill_between(x_vals, average_max_values[i] - std_max_values[i]/2, average_max_values[i] + std_max_values[i]/2,
                color=f"C{i}", alpha=0.3)
    line, = ax.plot(x_vals, average_max_values[i],
                    label = fr'$\lambda = {lamb}$', marker='o', markersize=5, alpha=0.7, color=f"C{i}")

    lines.append(line)

ax.fill_between(x_vals, average_max_values_fixed - std_max_values_fixed/2, average_max_values_fixed + std_max_values_fixed/2,
            color=f"C{len(lambs)}", alpha=0.3)

line_fixed, = ax.plot(x_vals, average_max_values_fixed,
                      label='Fixed', marker='o', markersize=5, alpha=0.7, color=f"C{len(lambs)}")

lines.append(line_fixed)

ax.set_yscale('log')
ax.set_xlim(0, 0.3)
ax.set_ylim(1e-4, 6e-2)
ax.legend(loc='upper left')
ax.set_xlabel('Noise Level')
ax.set_ylabel('MSE (worst case)')
ax.grid(linestyle='--', alpha=0.7)

# -------------------------------
# Create inset axes
inset_ax = inset_axes(ax, width="42%", height="42%", loc='lower right')  # loc can be: 'upper left', 'lower right', etc.

# Inset data (replace with your actual values)
x_labels = list(range(len(lambs) + 1))
values = [x for x in average_lip_bounds] + [average_lip_bounds_fixed]
values = [round(v, 2) for v in values]
colors = [line.get_color() for line in lines]

lip_errors = [std_lip_bounds[i] for i in range(len(lambs))] + [std_lip_bounds_fixed]
bars = inset_ax.bar(x_labels, values, color=colors, yerr=lip_errors, error_kw={
        'capsize': 5, 
    },)
for bar, value in zip(bars, values):
    inset_ax.text(bar.get_x() + bar.get_width() / 2, value * 1.2, f'{value}', ha='center', va='bottom')

inset_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
inset_ax.set_ylim(0, 55)
inset_ax.set_title("Lipschitz Bound", fontsize=10)
inset_ax.tick_params()

#plt.tight_layout()

plt.savefig(f"./Plots/mse_vs_noise_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}.pdf", bbox_inches='tight')