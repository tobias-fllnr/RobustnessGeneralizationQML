import torch
import matplotlib.pyplot as plt
from utils.analyzer import Analyzer
from utils.handling_data import DataHandling
from utils.circuit import Circuit
from utils.trainer import Trainer
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

# Use colorblind-friendly colors
color_train = "C0"  # Blue (CUD)
color_test = "C2"   # Orange (CUD)
color_baseline = "#666666"  # Neutral gray for baseline

fig, axes = plt.subplots(1, 3, figsize=(6, 2.1))

# Dictionary structure: {Title label: (trainable_encoding, random_id, lambda)}
settings = {
    r"$\lambda = 0.0$ trainable enc.": (True, 9, 0.0),
    r"$\lambda = 0.004$ trainable enc.": (True, 30, 0.004),
    r"fixed enc.": (False, 34, 0.0)
}

for i, (title, (trainable_encoding, random_id, lamb)) in enumerate(settings.items()):
    ax = axes[i]

    # Model setup
    model = Circuit(num_qubits=num_qubits, seq_length=seq_length, data_label=data_label, random_id=random_id, trainable_encoding=trainable_encoding)
    data_handler = DataHandling(data_label=data_label, seq_length=seq_length, prediction_step=prediction_step, random_id=random_id)
    trainer = Trainer(model=model, lamb=lamb, learning_rate=learning_rate, epochs=epochs)
    analyzer = Analyzer(version=version, model=model, data_handler=data_handler, trainer=trainer)

    # Load model
    model.load_state_dict(torch.load(analyzer.path + "/trained_model"))
    model.eval()

    # Get and rescale data
    inputs_train, labels_train, inputs_test, labels_test = data_handler.get_training_and_test_data()
    preds_train = 0.5 * model(inputs_train).detach().numpy() + 3.5
    preds_test = 0.5 * model(inputs_test).detach().numpy() + 3.5
    labels_train = 0.5 * labels_train.numpy() + 3.5
    labels_test = 0.5 * labels_test.numpy() + 3.5

    # Plot
    ax.scatter(labels_test, preds_test, alpha=0.5, label='Test', marker='.', color=color_test, s=15)
    ax.scatter(labels_train, preds_train, alpha=0.5, label='Train', marker='.', color=color_train, s=15)

    ax.plot([3.5, 4], [3.5, 4], linestyle='--', color=color_baseline, label='Baseline')

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("True Value")
    if i == 0:
        ax.set_ylabel("Predicted Value")
    ax.set_xlim(3.48, 4.02)
    ax.set_ylim(3.48, 4.02)
    ax.legend(fontsize=7, loc='lower right')
    #ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(f"./Plots/prediction_scatter_version_{version}_{data_label}_seq_{seq_length}_qubits_{num_qubits}.pdf", bbox_inches='tight')