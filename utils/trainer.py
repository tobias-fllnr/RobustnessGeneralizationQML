import torch
import time
import torch.nn as nn

class Trainer:
    def __init__(self,
                model,
                lamb,
                learning_rate=0.1, epochs=1000, print_gradients=False):
        super(Trainer, self).__init__()
        self.model = model
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_gradients = print_gradients
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cost = nn.MSELoss() 

    def train(self, inputs_training, labels_training, inputs_testing, labels_testing):
        time_start = time.time()
        cost_training = []
        mse_cost_training = []
        mse_cost_testing = []
        for epoch in range(self.epochs):
            time_epoch_start = time.time()
            self.optimizer.zero_grad()
            output_training = self.model(inputs_training)
            # Compute the original MSE loss
            original_loss = self.cost(output_training, labels_training)
            
            regularization = self.lamb * sum(
                ((p*0.5) ** 2).mean() 
                for name, p in self.model.named_parameters() 
                if name == "vqc_torch_layer.weights"  # <-- Filter by parameter name
            )
            loss_training = original_loss + regularization
            loss_training.backward()
            self.optimizer.step()

            if self.print_gradients:
                print(f"Gradients for epoch {epoch+1}:")
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"\t{name}:")
                        print(param.grad)
                    else:
                        print(f"\t{name}: No gradient")

            cost_training.append(loss_training.item())
            mse_cost_training.append(original_loss.item())
            with torch.no_grad():
                output_testing = self.model(inputs_testing)
                loss_testing = self.cost(output_testing, labels_testing)
                mse_cost_testing.append(loss_testing.item())
                
            time_epoch_end = time.time()
            print(f"Epoch {epoch+1}: mse loss training={round(mse_cost_training[-1], 7)},"
                  f"regularization={round(regularization.item(), 7)}," 
                  f"mse loss testing={round(mse_cost_testing[-1], 7)}," 
                  f"time for epoch={round(time_epoch_end-time_epoch_start, 2)}",
                  flush=True)

        time_end = time.time()
        total_time = round(time_end - time_start, 2)
        print(f"Training finished! Total time={total_time}", flush=True)
        model_end = self.model.state_dict()
        return cost_training, mse_cost_training, mse_cost_testing, model_end

    