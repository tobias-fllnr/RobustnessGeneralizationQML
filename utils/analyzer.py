import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")


class Analyzer:
    """
    class to analyse the model after training
    """
    def __init__(self, version, model, trainer, data_handler):
        self.model = model
        self.trainer = trainer
        self.data_handler = data_handler
        self.path = f"./Data/Version_{version}/{self.data_handler.data_label}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/Lambda_{self.trainer.lamb}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_epochs_{self.trainer.epochs}_Trainable_Encoding_{self.model.trainable_encoding}"
    def create_directory(self):
        """
        create the directory to store all data and plots of the specific training
        """
        # Check if the directory already exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            pass

    def load_model(self):
        """
        checks if directory exists. If exists, return true and load model, if not return false
        :return:
        """
        if os.path.exists(self.path + "/trained_model"):
            self.model.load_state_dict(torch.load(self.path + "/trained_model", weights_only=True))
            self.model.eval()
            return True
        else:
            return False
    
    def save_training_output(self,trained_model, cost_training, mse_cost_training, mse_cost_testing):
        """
        save outputs of training
        """
        torch.save(trained_model, self.path + "/trained_model")
        np.save(self.path + "/cost_training", cost_training)
        np.save(self.path + "/mse_cost_training", mse_cost_training)
        np.save(self.path + "/mse_cost_testing", mse_cost_testing)

    def plot_cost(self, cost_training, mse_cost_training, mse_cost_testing):
        """
        plot cost over epochs
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
    
    def evaluate_trained_model(self, inputs_testing, labels_testing):
        """
        evaluate the trained model
        """

        def calculate_values(output_testing):
            mse_testing = nn.MSELoss()(output_testing, labels_testing).item()
            mae_testing = nn.L1Loss()(output_testing, labels_testing).item()

            corr_test = torch.stack((output_testing, labels_testing))
            corr_test = torch.reshape(corr_test, (corr_test.size(0), -1))
            corr_testing = torch.corrcoef(corr_test)[0][1].item()

            lipschitz_bound = 2* sum(
                ((p*0.5) ** 2).sum() 
                for name, p in self.model.named_parameters() 
                if name == "vqc_torch_layer.weights"  # <-- Filter by parameter name
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

        
    def get_number_of_parameters(self):
        """
        get the number of parameters of the model
        """
        total_parameters = 0
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                total_parameters += param.numel()
        return total_parameters
    

    def evaluate_with_test_noise(self, inputs_testing, labels_testing):
        """
        Evaluate the model with test noise
        """

        num_noise_levels = 31
        num_noise_samples = 10
        max_noise_level = 0.3
        
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

    
    def plot_r_prediction(self, inputs_training, labels_training, inputs_testing, labels_testing):
        """
        plot the predictions of the model
        """
        predictions_training = self.model(inputs_training)
        predictions_testing = self.model(inputs_testing)
        if self.data_handler.data_label == "logistic_map_1000_chaos":
            plt.scatter(0.5*labels_training.numpy()+3.5, 0.5*predictions_training.detach().numpy()+3.5, alpha=0.5, label='Training Data', marker='.')
            plt.scatter(0.5*labels_testing.numpy()+3.5, 0.5*predictions_testing.detach().numpy()+3.5, alpha=0.5, label='Test Data', marker='.')
            plt.plot([3.5, 4], [3.5, 4], color='red', linestyle='--', label='Baseline')
        else:
            plt.scatter(4*labels_training.numpy(), 4*predictions_training.detach().numpy(), alpha=0.5, label='Training Data', marker='.')
            plt.scatter(4*labels_testing.numpy(), 4*predictions_testing.detach().numpy(), alpha=0.5, label='Test Data', marker='.')
            plt.plot([0, 4], [0, 4], color='red', linestyle='--', label='Baseline')

        plt.xlabel("True Labels")
        plt.ylabel("Predictions")
        plt.title("Scatter Plot of Predictions vs True Labels")

        plt.legend()
        plt.savefig(self.path + "/plot_predictions_r.pdf")