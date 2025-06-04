import pandas as pd
import torch
import numpy as np

class DataHandling:
    """Class to handle time series data loading, normalization, and splitting for training/testing."""

    def __init__(self, data_label, seq_length, prediction_step, random_id):
        """
        Initialize DataHandling class.

        Parameters:
        - data_label_entry: Identifier for the dataset to be used.
        - seq_length: Length of input sequences for the model.
        - prediction_step: How many time steps ahead the model should predict.
        """

        self.data_label = data_label
        self.seq_length = seq_length
        self.prediction_step = prediction_step
        self.random_id = random_id

        # Predefined dataset metadata: file paths, lengths, and split sizes
        data_info = {
            "logistic_map_1000_chaos": {"file_path": "./TimeseriesData/logistic_map_chaos.csv", "data_length": 1000, "test_size": 0.8}
        }

        # Load dataset metadata if label is found
        if self.data_label in data_info:
            self.file_path = data_info[self.data_label]["file_path"]
            self.data_length = data_info[self.data_label]["data_length"]
            self.test_size = data_info[self.data_label]["test_size"]
        else:
            raise ValueError("Data label not found in data_info")

        # Load the actual data and keep min/max values for normalization
        if not self.data_label.startswith("logistic_map"): 
            self.data, self.min_values, self.max_values = self.load_data()
    
    def load_data(self):
        """
        Load data from CSV file and compute min/max values per column.

        Returns:
        - data: Pandas DataFrame containing the loaded data
        - min_values: List of minimum values per column (for normalization)
        - max_values: List of maximum values per column
        """
        data = pd.read_csv(self.file_path)
        data = data.head(self.data_length)  # Truncate to specified length
        min_values = []
        max_values = []
        for column in data.columns:
            min_values.append(data[column].min())
            max_values.append(data[column].max())
        return data, min_values, max_values

    def transform(self):
        """
        Normalize each column in the data to the [0, 1] range.

        Returns:
        - Normalized data (Pandas DataFrame)
        """
        data = self.data.copy()
        for i, column in enumerate(self.data.columns):
            data[column] = (data[column] - self.min_values[i]) / (self.max_values[i] - self.min_values[i])
        return data
    
    def inverse_transform(self, data):
        """
        Revert the normalization back to original data scale.

        Parameters:
        - data: Pandas DataFrame with normalized values

        Returns:
        - DataFrame with original scale values
        """
        for i, column in enumerate(data.columns):
            data[column] = data[column] * (self.max_values[i] - self.min_values[i]) + self.min_values[i]
        return data

    def get_training_and_test_data(self):
        """
        Split the time series into training, validation, and test sets using sliding windows.

        Returns:
        - inputs_training: Tensor of input sequences for training
        - labels_training: Tensor of corresponding target values
        - inputs_testing: Tensor for testing input sequences
        - labels_testing: Corresponding targets
        """

        if self.data_label.startswith("logistic_map"):
            data = pd.read_csv(self.file_path)
            data = data.head(self.seq_length)
            x = [data[col].values for col in data.columns]
            y = [float(col) for col in data.columns]
            min_val = min(y)
            max_val = max(y)
            y = [[(i - min_val) / (max_val - min_val)] for i in y]

        else:
            data = self.transform()
            x = []  # List to hold input sequences
            y = []  # List to hold corresponding targets

            # Construct input-output pairs with sliding window
            for i in range(len(data) - self.seq_length - self.prediction_step):
                x.append(data.iloc[i:i+self.seq_length].values)  # Input sequence
                y.append(data.iloc[i+self.seq_length+self.prediction_step-1].values)  # Prediction target

        # Calculate indices to split into train/validation/test
        split_index_test = int(len(x) * (1 - self.test_size))

        # Convert to NumPy arrays
        x, y = np.array(x), np.array(y)

        # Generate a permutation of indices
        # np.random.seed(self.random_id)  # For reproducibility
        # I want the same training and testing data for all models
        np.random.seed(42)
        p = np.random.permutation(len(x))

        # Shuffle both arrays with the same permutation
        x = x[p]
        y = y[p]

        # Convert to PyTorch tensors and split according to calculated indices
        inputs_training = torch.tensor(x[:split_index_test], dtype=torch.float32)
        inputs_testing = torch.tensor(x[split_index_test:], dtype=torch.float32)

        labels_training = torch.tensor(y[:split_index_test], dtype=torch.float32)
        labels_testing = torch.tensor(y[split_index_test:], dtype=torch.float32)

        return inputs_training, labels_training, inputs_testing, labels_testing
