import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Use default LaTeX font
    }
)


version = 13
learning_rate = 0.01
data_label = "logistic_map_1000_chaos"
seq_length = 12
prediction_step = 1
num_qubits = 4
epochs = 2000

random_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
lambs = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03]

def load_data(lamb, random_id, trainable_encoding):
    path =f"./Data/Version_{version}/{data_label}/Sequence_length_{seq_length}/Prediction_step_{prediction_step}/Num_qubits_{num_qubits}/Lambda_{lamb}/ID_{random_id}_lr_{learning_rate}_epochs_{epochs}_Trainable_Encoding_{trainable_encoding}"
    loss_metrics = pd.read_csv(path + "/loss_metrics_end.csv")
    lip_bound = loss_metrics["Lipschitz Bound"][0]
    mse_test_noise = np.load(path + "/mse_testing_noise.npy")
    max_values = np.max(mse_test_noise, axis=1)
    mse_train_loss = np.load(path + "/mse_cost_training.npy")
    mse_train_loss = mse_train_loss[-1]
    return lip_bound, max_values, mse_train_loss

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

max_lip_bounds = np.max(lip_bounds, axis=1)
max_max_values = np.max(max_values_array, axis=1)


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

max_lip_bounds_fixed = np.max(lip_bounds_fixed, axis=0)
max_max_values_fixed = np.max(max_values_array_fixed, axis=0)


fig, ax1 = plt.subplots(figsize=(5.2, 3.6))

# Left Y-axis: MSE (or Accuracy)
ax1.plot(lambs, average_max_values[:, 0], 'D-', color="#1c5d99", label='MSE test', alpha=0.8)
ax1.fill_between(lambs, average_max_values[:, 0] - std_max_values[:, 0]/2, average_max_values[:, 0] + std_max_values[:, 0]/2, alpha=0.2, color="#1c5d99")

ax1.plot(lambs, average_mse_train_loss, 'v-', color="#3a77b2", label='MSE train', alpha=0.8)
ax1.fill_between(lambs, average_mse_train_loss - std_mse_train_loss/2, average_mse_train_loss + std_mse_train_loss/2, alpha=0.2, color="#3a77b2")

ax1.plot(lambs, average_max_values[:, 0]-average_mse_train_loss, 'h-', color="#6aaee0", label='Generalization gap', alpha=0.8)
ax1.fill_between(lambs, average_max_values[:, 0]-average_mse_train_loss - np.sqrt(std_max_values[:, 0]**2 + std_mse_train_loss**2)/2, average_max_values[:, 0]-average_mse_train_loss + np.sqrt(std_max_values[:, 0]**2 + std_mse_train_loss**2)/2, alpha=0.2, color="#6aaee0")

ax1.set_ylabel('MSE', color="C0")
#ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor="C0")
#ax1.set_ylim(1e-4, 8e-4)  # Adjust based on your data
ax1.grid(axis='x', linestyle='--', alpha=0.7) 

def scientific_formatter(x, pos):
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    # Format coefficient to 1 decimal place (optional)
    return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)

ax1.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
#ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax1.yaxis.get_offset_text().set_fontsize(10)  # Optional: make the offset text smaller

# Right Y-axis: Lipschitz Bound
ax2 = ax1.twinx()
ax2.plot(lambs, average_lip_bounds, '*-', color="C2", label='Lipschitz Bound', alpha=0.8)
ax2.fill_between(lambs, average_lip_bounds - std_lip_bounds/2, average_lip_bounds + std_lip_bounds/2, alpha=0.2, color="C2")
ax2.set_ylabel('Lipschitz Bound', color="C2")
#ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor="C2")
#ax2.set_ylim([1, 50])  # Adjust based on your data

# Common X-axis
ax1.set_xlabel('Regularization parameter $\\lambda$')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=(3,3), fontsize=9)
plt.tight_layout()
plt.savefig(f'./Plots/mse_vs_lambda_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}_generalization_gap.pdf')
