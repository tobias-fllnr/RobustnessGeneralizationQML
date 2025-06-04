import pennylane as qml
import torch
import torch.nn as nn

class Circuit(nn.Module):  # Fixed typo: 'nn.module' → 'nn.Module'
    def __init__(self, num_qubits, seq_length, data_label, random_id, trainable_encoding=True):
        super().__init__()
        self.num_qubits = num_qubits
        self.seq_length = seq_length
        self.data_label = data_label
        self.random_id = random_id
        self.trainable_encoding = trainable_encoding
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.diff_method = "best"

        # Set seed for reproducibility
        torch.manual_seed(self.random_id)

        # Define initial weight initialization method
        self.weight_init = {"weights": lambda x: torch.nn.init.uniform_(x, -torch.pi/2, torch.pi/2), 
                            "bias": lambda x: torch.nn.init.uniform_(x, -torch.pi/2, torch.pi/2)}

        # Determine input dimension based on data label
        if self.data_label.startswith("lorenz"):
            self.data_dim = 3
        elif self.data_label.startswith("henon"):
            self.data_dim = 2
        elif self.data_label.startswith("mackey"):
            self.data_dim = 1
        elif self.data_label.startswith("logistic_map"):
            self.data_dim = 1
        else:
            raise ValueError(f"Unsupported data label: {self.data_label}")

        # Output layer to map quantum outputs to target dimension
        self.output_layer = nn.Linear(self.num_qubits, self.data_dim)

        # Create variational quantum circuit (as a TorchLayer)
        self.vqc_torch_layer = self.vqc()

        # Set trainable parameters
        self.vqc_torch_layer.weights.requires_grad = self.trainable_encoding  # Set to False to freeze
        self.vqc_torch_layer.bias.requires_grad = True     # Set to False to freeze

    def vqc(self):
        @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
        def circuit(inputs, weights, bias):
            for i in range(self.seq_length):
                for j in range(self.num_qubits):
                    if self.data_dim == 1:
                        dim1 = weights[i][j][0] * inputs[:, i] + bias[i][j][0]
                        dim2 = weights[i][j][1] * inputs[:, i] + bias[i][j][1]
                    elif self.data_dim == 2:
                        dim1 = weights[i][j][0] * inputs[:, 2*i] + weights[i][j][1] * inputs[:, 2*i+1] + bias[i][j][0]
                        dim2 = weights[i][j][2] * inputs[:, 2*i] + weights[i][j][3] * inputs[:, 2*i+1] + bias[i][j][1]
                    elif self.data_dim == 3:
                        dim1 = weights[i][j][0] * inputs[:, 3*i] + weights[i][j][1] * inputs[:, 3*i+1] + weights[i][j][2] * inputs[:, 3*i+2] + bias[i][j][0]
                        dim2 = weights[i][j][3] * inputs[:, 3*i] + weights[i][j][4] * inputs[:, 3*i+1] + weights[i][j][5] * inputs[:, 3*i+2] + bias[i][j][1]
                    else:
                        raise ValueError(f"Unsupported data dimension: {self.data_dim}")

                    qml.RY(dim1, wires=j)
                    qml.RZ(dim2, wires=j)

                # Apply entangling CNOT gates
                if self.num_qubits >= 2:
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                if self.num_qubits >= 3:
                    qml.CNOT(wires=[self.num_qubits - 1, 0])

            # Measure expectation values of PauliZ on each qubit
            if self.data_dim == 1:
                return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits)]))]
            elif self.data_dim == 2:
                return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits//2)])), qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits//2, self.num_qubits)]))]
            elif self.data_dim == 3:
                return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits//3)])), 
                        qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits//3, 2*self.num_qubits//3)])), 
                        qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(2*self.num_qubits//3, self.num_qubits)]))]
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        weight_shapes = {
            "weights": (self.seq_length, self.num_qubits, 2 * self.data_dim),
            "bias": (self.seq_length, self.num_qubits, 2),
        }

        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)

    def forward(self, x):
        # Flatten input per sample (batch dimension is preserved)
        x = torch.reshape(x, (x.size(0), -1))

        # Forward through VQC
        output = self.vqc_torch_layer(x)
        # Renormalizwe output from [-1, 1] to [0, 1]
        output = output + 1 
        output = output / 2


        # Linear transformation to match output dimension
        #output = self.output_layer(vqc_output)
        return output
