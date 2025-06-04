import pennylane as qml
import torch
import torch.nn as nn

class Circuit(nn.Module):
    """
    A class representing a variational quantum circuit (VQC) implemented using PennyLane and PyTorch.

    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit.
        seq_length (int): Sequence length for the input data.
        data_label (str): Label describing the type of data being processed.
        random_id (int): Random seed for reproducibility.
        trainable_encoding (bool): Whether the encoding weights are trainable.
        dev (qml.Device): Quantum device used for simulation.
        diff_method (str): Differentiation method for the quantum circuit.
        weight_init (dict): Dictionary containing weight initialization methods.
        data_dim (int): Dimensionality of the input data.
        output_layer (nn.Linear): Linear layer to map quantum outputs to target dimensions.
        vqc_torch_layer (qml.qnn.TorchLayer): Variational quantum circuit wrapped as a TorchLayer.
    """

    def __init__(self, num_qubits: int, seq_length: int, data_label: str, random_id: int, trainable_encoding: bool = True):
        """
        Initializes the Circuit class.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            seq_length (int): Sequence length for the input data.
            data_label (str): Label describing the type of data being processed.
            random_id (int): Random seed for reproducibility.
            trainable_encoding (bool, optional): Whether the encoding weights are trainable. Defaults to True.
        """
        super().__init__()
        self.num_qubits: int = num_qubits
        self.seq_length: int = seq_length
        self.data_label: str = data_label
        self.random_id: int = random_id
        self.trainable_encoding: bool = trainable_encoding
        self.dev: qml.Device = qml.device("default.qubit", wires=self.num_qubits)
        self.diff_method: str = "best"

        # Set seed for reproducibility
        torch.manual_seed(self.random_id)

        # Define initial weight initialization method
        self.weight_init: dict = {
            "weights": lambda x: torch.nn.init.uniform_(x, -torch.pi / 2, torch.pi / 2),
            "bias": lambda x: torch.nn.init.uniform_(x, -torch.pi / 2, torch.pi / 2),
        }

        # Determine input dimension based on data label
        if self.data_label.startswith("logistic_map"):
            self.data_dim: int = 1
        else:
            raise ValueError(f"Unsupported data label: {self.data_label}")

        # Output layer to map quantum outputs to target dimension
        self.output_layer: nn.Linear = nn.Linear(self.num_qubits, self.data_dim)

        # Create variational quantum circuit (as a TorchLayer)
        self.vqc_torch_layer: qml.qnn.TorchLayer = self.vqc()

        # Set trainable parameters
        self.vqc_torch_layer.weights.requires_grad = self.trainable_encoding
        self.vqc_torch_layer.bias.requires_grad = True

    def vqc(self) -> qml.qnn.TorchLayer:
        """
        Defines the variational quantum circuit (VQC).

        Returns:
            qml.qnn.TorchLayer: The VQC wrapped as a TorchLayer.
        """
        @qml.qnode(self.dev, diff_method=self.diff_method, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> list:
            """
            Quantum circuit definition.

            Args:
                inputs (torch.Tensor): Input data tensor.
                weights (torch.Tensor): Weights tensor for the circuit.
                bias (torch.Tensor): Bias tensor for the circuit.

            Returns:
                list: Expectation values of PauliZ measurements.
            """
            for i in range(self.seq_length):
                for j in range(self.num_qubits):
                    if self.data_dim == 1:
                        dim1 = weights[i][j][0] * inputs[:, i] + bias[i][j][0]
                        dim2 = weights[i][j][1] * inputs[:, i] + bias[i][j][1]
                    elif self.data_dim == 2:
                        dim1 = weights[i][j][0] * inputs[:, 2 * i] + weights[i][j][1] * inputs[:, 2 * i + 1] + bias[i][j][0]
                        dim2 = weights[i][j][2] * inputs[:, 2 * i] + weights[i][j][3] * inputs[:, 2 * i + 1] + bias[i][j][1]
                    elif self.data_dim == 3:
                        dim1 = weights[i][j][0] * inputs[:, 3 * i] + weights[i][j][1] * inputs[:, 3 * i + 1] + weights[i][j][2] * inputs[:, 3 * i + 2] + bias[i][j][0]
                        dim2 = weights[i][j][3] * inputs[:, 3 * i] + weights[i][j][4] * inputs[:, 3 * i + 1] + weights[i][j][5] * inputs[:, 3 * i + 2] + bias[i][j][1]
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
                return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits // 2)])),
                        qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits // 2, self.num_qubits)]))]
            elif self.data_dim == 3:
                return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits // 3)])),
                        qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(self.num_qubits // 3, 2 * self.num_qubits // 3)])),
                        qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(2 * self.num_qubits // 3, self.num_qubits)]))]

        weight_shapes: dict = {
            "weights": (self.seq_length, self.num_qubits, 2 * self.data_dim),
            "bias": (self.seq_length, self.num_qubits, 2),
        }

        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the quantum circuit.
        """
        # Flatten input per sample (batch dimension is preserved)
        x = torch.reshape(x, (x.size(0), -1))

        # Forward through VQC
        output = self.vqc_torch_layer(x)

        # Renormalize output from [-1, 1] to [0, 1]
        output = output + 1
        output = output / 2

        return output
