import pandas as pd
import torch
import numpy as np

class DataHandling:
    """
    Class to handle time series data loading, normalization, and splitting for training/testing.

    Attributes:
        data_label (str): Identifier for the dataset to be used.
        seq_length (int): Length of input sequences for the model.
        prediction_step (int): How many time steps ahead the model should predict.
        random_id (int): Random seed for reproducibility.
        file_path (str): Path to the dataset file.
        data_length (int): Length of the dataset to be used.
        test_size (float): Fraction of data to be used for testing.
        data (pd.DataFrame): Loaded dataset.
        min_values (list): Minimum values per column for normalization.
        max_values (list): Maximum values per column for normalization.
    """

    def __init__(self, data_label: str, seq_length: int, prediction_step: int, random_id: int):
        """
        Initializes the DataHandling class.

        Args:
            data_label (str): Identifier for the dataset to be used.
            seq_length (int): Length of input sequences for the model.
            prediction_step (int): How many time steps ahead the model should predict.
            random_id (int): Random seed for reproducibility.

        Raises:
            ValueError: If the data label is not found in predefined metadata.
        """
        self.data_label: str = data_label
        self.seq_length: int = seq_length
        self.prediction_step: int = prediction_step
        self.random_id: int = random_id

        # Predefined dataset metadata: file paths, lengths, and split sizes
        data_info: dict = {
            "logistic_map_1000_chaos": {
                "file_path": "./TimeseriesData/logistic_map_chaos.csv",
                "data_length": 1000,
                "test_size": 0.8,
            }
        }

        # Load dataset metadata if label is found
        if self.data_label in data_info:
            self.file_path: str = data_info[self.data_label]["file_path"]
            self.data_length: int = data_info[self.data_label]["data_length"]
            self.test_size: float = data_info[self.data_label]["test_size"]
        else:
            raise ValueError("Data label not found in data_info")

        # Load the actual data and keep min/max values for normalization
        if not self.data_label.startswith("logistic_map"):
            self.data, self.min_values, self.max_values = self.load_data()

    def load_data(self) -> tuple:
        """
        Loads data from a CSV file and computes min/max values per column.

        Returns:
            tuple: A tuple containing:
                - data (pd.DataFrame): Loaded dataset.
                - min_values (list): Minimum values per column for normalization.
                - max_values (list): Maximum values per column for normalization.
        """
        data: pd.DataFrame = pd.read_csv(self.file_path)
        data = data.head(self.data_length)  # Truncate to specified length
        min_values: list = []
        max_values: list = []
        for column in data.columns:
            min_values.append(data[column].min())
            max_values.append(data[column].max())
        return data, min_values, max_values

    def transform(self) -> pd.DataFrame:
        """
        Normalizes each column in the data to the [0, 1] range.

        Returns:
            pd.DataFrame: Normalized data.
        """
        data: pd.DataFrame = self.data.copy()
        for i, column in enumerate(self.data.columns):
            data[column] = (data[column] - self.min_values[i]) / (self.max_values[i] - self.min_values[i])
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverts the normalization back to the original data scale.

        Args:
            data (pd.DataFrame): DataFrame with normalized values.

        Returns:
            pd.DataFrame: DataFrame with original scale values.
        """
        for i, column in enumerate(data.columns):
            data[column] = data[column] * (self.max_values[i] - self.min_values[i]) + self.min_values[i]
        return data

    def get_training_and_test_data(self) -> tuple:
        """
        Splits the time series into training, validation, and test sets using sliding windows.

        Returns:
            tuple: A tuple containing:
                - inputs_training (torch.Tensor): Tensor of input sequences for training.
                - labels_training (torch.Tensor): Tensor of corresponding target values for training.
                - inputs_testing (torch.Tensor): Tensor of input sequences for testing.
                - labels_testing (torch.Tensor): Tensor of corresponding target values for testing.
        """

        data: pd.DataFrame = pd.read_csv(self.file_path)
        data = data.head(self.seq_length)
        x: list = [data[col].values for col in data.columns]
        y: list = [float(col) for col in data.columns]
        min_val: float = min(y)
        max_val: float = max(y)
        y = [[(i - min_val) / (max_val - min_val)] for i in y]


        # Calculate indices to split into train/validation/test
        split_index_test: int = int(len(x) * (1 - self.test_size))

        # Convert to NumPy arrays
        x: np.ndarray = np.array(x)
        y: np.ndarray = np.array(y)

        # Generate a permutation of indices
        np.random.seed(42)  # For reproducibility
        p: np.ndarray = np.random.permutation(len(x))

        # Shuffle both arrays with the same permutation
        x = x[p]
        y = y[p]

        # Convert to PyTorch tensors and split according to calculated indices
        inputs_training: torch.Tensor = torch.tensor(x[:split_index_test], dtype=torch.float32)
        inputs_testing: torch.Tensor = torch.tensor(x[split_index_test:], dtype=torch.float32)

        labels_training: torch.Tensor = torch.tensor(y[:split_index_test], dtype=torch.float32)
        labels_testing: torch.Tensor = torch.tensor(y[split_index_test:], dtype=torch.float32)

        return inputs_training, labels_training, inputs_testing, labels_testing
