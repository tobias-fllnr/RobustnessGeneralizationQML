import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")


class Analyzer:
    """
    A class to analyze the model after training.

    Attributes:
        model (torch.nn.Module): The trained model.
        trainer (object): The trainer object containing training parameters.
        data_handler (object): The data handler object containing data-related parameters.
        path (str): The directory path to store data and plots for the specific training.
    """

    def __init__(self, version: str, model: nn.Module, trainer: object, data_handler: object):
        """
        Initializes the Analyzer class.

        Args:
            version (str): Version identifier for the training.
            model (torch.nn.Module): The trained model.
            trainer (object): The trainer object containing training parameters.
            data_handler (object): The data handler object containing data-related parameters.
        """
        self.model: nn.Module = model
        self.trainer: object = trainer
        self.data_handler: object = data_handler
        self.path: str = (
            f"./Data/Version_{version}/{self.data_handler.data_label}/Sequence_length_{self.data_handler.seq_length}/"
            f"Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/Lambda_{self.trainer.lamb}/"
            f"ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_epochs_{self.trainer.epochs}_Trainable_Encoding_{self.model.trainable_encoding}"
        )

    def create_directory(self) -> None:
        """
        Creates the directory to store all data and plots of the specific training.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def load_model(self) -> bool:
        """
        Loads the trained model if the directory exists.

        Returns:
            bool: True if the model is loaded successfully, False otherwise.
        """
        if os.path.exists(self.path + "/trained_model"):
            self.model.load_state_dict(torch.load(self.path + "/trained_model", weights_only=True))
            self.model.eval()
            return True
        return False

    def save_training_output(self, trained_model: nn.Module, cost_training: np.ndarray, mse_cost_training: np.ndarray, mse_cost_testing: np.ndarray) -> None:
        """
        Saves the outputs of training.

        Args:
            trained_model (torch.nn.Module): The trained model.
            cost_training (np.ndarray): Cost values during training.
            mse_cost_training (np.ndarray): Mean squared error during training.
            mse_cost_testing (np.ndarray): Mean squared error during testing.
        """
        torch.save(trained_model, self.path + "/trained_model")
        np.save(self.path + "/cost_training", cost_training)
        np.save(self.path + "/mse_cost_training", mse_cost_training)
        np.save(self.path + "/mse_cost_testing", mse_cost_testing)

    def plot_cost(self, cost_training: np.ndarray, mse_cost_training: np.ndarray, mse_cost_testing: np.ndarray) -> None:
        """
        Plots the cost over epochs.

        Args:
            cost_training (np.ndarray): Cost values during training.
            mse_cost_training (np.ndarray): Mean squared error during training.
            mse_cost_testing (np.ndarray): Mean squared error during testing.
        """
        plt.figure()
        plt.plot(cost_training, label="Training")
        plt.plot(mse_cost_training, label="MSE Training")
        plt.plot(mse_cost_testing, label="MSE Testing")
        plt.xlabel("Epoch")
        plt.ylabel("Cost (MSE (+regularization))")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.path + "/cost_plot.pdf")
        plt.close()

    def evaluate_trained_model(self, inputs_testing: torch.Tensor, labels_testing: torch.Tensor) -> None:
        """
        Evaluates the trained model.

        Args:
            inputs_testing (torch.Tensor): Testing inputs.
            labels_testing (torch.Tensor): Testing labels.
        """
        def calculate_values(output_testing: torch.Tensor) -> pd.DataFrame:
            """
            Calculates evaluation metrics.

            Args:
                output_testing (torch.Tensor): Model predictions for testing data.

            Returns:
                pd.DataFrame: DataFrame containing evaluation metrics.
            """
            mse_testing = nn.MSELoss()(output_testing, labels_testing).item()
            mae_testing = nn.L1Loss()(output_testing, labels_testing).item()

            corr_test = torch.stack((output_testing, labels_testing))
            corr_test = torch.reshape(corr_test, (corr_test.size(0), -1))
            corr_testing = torch.corrcoef(corr_test)[0][1].item()

            lipschitz_bound = 2 * sum(
                ((p * 0.5) ** 2).sum()
                for name, p in self.model.named_parameters()
                if name == "vqc_torch_layer.weights"
            )

            data = {
                "MSE Testing": mse_testing,
                "MAE Testing": mae_testing,
                "Correlation Testing": corr_testing,
                "Lipschitz Bound": lipschitz_bound.item(),
            }
            return pd.DataFrame(data, index=[0])

        self.model.load_state_dict(torch.load(self.path + "/trained_model", weights_only=True))
        self.model.eval()
        output_testing = self.model(inputs_testing)
        df_last = calculate_values(output_testing)
        df_last.to_csv(self.path + "/loss_metrics_end.csv", index=False)

    def get_number_of_parameters(self) -> int:
        """
        Gets the number of parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        total_parameters = sum(param.numel() for _, param in self.model.named_parameters() if param.requires_grad)
        return total_parameters

    def evaluate_with_test_noise(self, inputs_testing: torch.Tensor, labels_testing: torch.Tensor) -> None:
        """
        Evaluates the model with test noise.

        Args:
            inputs_testing (torch.Tensor): Testing inputs.
            labels_testing (torch.Tensor): Testing labels.
        """
        num_noise_levels = 51
        num_noise_samples = 100
        max_noise_level = 0.5

        torch.manual_seed(42)

        noise_levels = torch.linspace(0, max_noise_level, num_noise_levels)
        results = torch.zeros(num_noise_levels, num_noise_samples)
        mse_loss = nn.MSELoss()

        inputs_testing = inputs_testing.to(dtype=torch.float32)
        labels_testing = labels_testing.to(dtype=torch.float32)

        for i, noise_level in enumerate(noise_levels):
            for j in range(num_noise_samples):
                noise = 2 * noise_level * torch.rand_like(inputs_testing) - noise_level
                noisy_inputs = inputs_testing + noise
                with torch.no_grad():
                    output = self.model(noisy_inputs)
                    loss = mse_loss(output, labels_testing)
                results[i, j] = loss

        np.save(self.path + "/mse_testing_noise", results.numpy())

    def plot_r_prediction(self, inputs_training: torch.Tensor, labels_training: torch.Tensor, inputs_testing: torch.Tensor, labels_testing: torch.Tensor) -> None:
        """
        Plots the predictions of the model.

        Args:
            inputs_training (torch.Tensor): Training inputs.
            labels_training (torch.Tensor): Training labels.
            inputs_testing (torch.Tensor): Testing inputs.
            labels_testing (torch.Tensor): Testing labels.
        """
        predictions_training = self.model(inputs_training)
        predictions_testing = self.model(inputs_testing)
        if self.data_handler.data_label == "logistic_map_1000_chaos":
            plt.scatter(0.5 * labels_training.numpy() + 3.5, 0.5 * predictions_training.detach().numpy() + 3.5, alpha=0.5, label='Training Data', marker='.')
            plt.scatter(0.5 * labels_testing.numpy() + 3.5, 0.5 * predictions_testing.detach().numpy() + 3.5, alpha=0.5, label='Test Data', marker='.')
            plt.plot([3.5, 4], [3.5, 4], color='red', linestyle='--', label='Baseline')
        else:
            plt.scatter(4 * labels_training.numpy(), 4 * predictions_training.detach().numpy(), alpha=0.5, label='Training Data', marker='.')
            plt.scatter(4 * labels_testing.numpy(), 4 * predictions_testing.detach().numpy(), alpha=0.5, label='Test Data', marker='.')
            plt.plot([0, 4], [0, 4], color='red', linestyle='--', label='Baseline')

        plt.xlabel("True Labels")
        plt.ylabel("Predictions")
        plt.title("Scatter Plot of Predictions vs True Labels")

        plt.legend()
        plt.savefig(self.path + "/plot_predictions_r.pdf")