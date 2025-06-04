import torch
import time
import torch.nn as nn

class Trainer:
    """
    A class to train a PyTorch model with regularization.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        lamb (float): Regularization coefficient.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        print_gradients (bool): Whether to print gradients during training.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        cost (torch.nn.Module): Loss function (Mean Squared Error).
    """

    def __init__(self, model: nn.Module, lamb: float, learning_rate: float = 0.1, epochs: int = 1000, print_gradients: bool = False):
        """
        Initializes the Trainer class.

        Args:
            model (torch.nn.Module): The model to be trained.
            lamb (float): Regularization coefficient.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            epochs (int, optional): Number of training epochs. Defaults to 1000.
            print_gradients (bool, optional): Whether to print gradients during training. Defaults to False.
        """
        super(Trainer, self).__init__()
        self.model: nn.Module = model
        self.lamb: float = lamb
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.print_gradients: bool = print_gradients
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cost: nn.Module = nn.MSELoss()

    def train(self, inputs_training: torch.Tensor, labels_training: torch.Tensor, inputs_testing: torch.Tensor, labels_testing: torch.Tensor) -> tuple:
        """
        Trains the model using the provided training and testing data.

        Args:
            inputs_training (torch.Tensor): Input data for training.
            labels_training (torch.Tensor): Target labels for training.
            inputs_testing (torch.Tensor): Input data for testing.
            labels_testing (torch.Tensor): Target labels for testing.

        Returns:
            tuple: A tuple containing:
                - cost_training (list): List of training loss values (MSE + regularization) per epoch.
                - mse_cost_training (list): List of training MSE loss values per epoch.
                - mse_cost_testing (list): List of testing MSE loss values per epoch.
                - model_end (dict): State dictionary of the trained model.
        """
        time_start: float = time.time()
        cost_training: list = []
        mse_cost_training: list = []
        mse_cost_testing: list = []

        for epoch in range(self.epochs):
            time_epoch_start: float = time.time()
            self.optimizer.zero_grad()

            # Forward pass for training data
            output_training: torch.Tensor = self.model(inputs_training)

            # Compute the original MSE loss
            original_loss: torch.Tensor = self.cost(output_training, labels_training)

            # Compute regularization term
            regularization: torch.Tensor = self.lamb * sum(
                ((p * 0.5) ** 2).mean()
                for name, p in self.model.named_parameters()
                if name == "vqc_torch_layer.weights"  # Filter by parameter name
            )

            # Compute total training loss
            loss_training: torch.Tensor = original_loss + regularization

            # Backpropagation and optimization
            loss_training.backward()
            self.optimizer.step()

            # Optionally print gradients
            if self.print_gradients:
                print(f"Gradients for epoch {epoch + 1}:")
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"\t{name}:")
                        print(param.grad)
                    else:
                        print(f"\t{name}: No gradient")

            # Record training loss
            cost_training.append(loss_training.item())
            mse_cost_training.append(original_loss.item())

            # Evaluate on testing data
            with torch.no_grad():
                output_testing: torch.Tensor = self.model(inputs_testing)
                loss_testing: torch.Tensor = self.cost(output_testing, labels_testing)
                mse_cost_testing.append(loss_testing.item())

            time_epoch_end: float = time.time()
            print(f"Epoch {epoch + 1}: mse loss training={round(mse_cost_training[-1], 7)},"
                  f" regularization={round(regularization.item(), 7)},"
                  f" mse loss testing={round(mse_cost_testing[-1], 7)},"
                  f" time for epoch={round(time_epoch_end - time_epoch_start, 2)}",
                  flush=True)

        time_end: float = time.time()
        total_time: float = round(time_end - time_start, 2)
        print(f"Training finished! Total time={total_time}", flush=True)

        model_end: dict = self.model.state_dict()
        return cost_training, mse_cost_training, mse_cost_testing, model_end