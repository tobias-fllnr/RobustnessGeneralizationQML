from handling_data import DataHandling
import circuit
from trainer import Trainer
from analyzer import Analyzer
import time
import argparse
import ast

def train_and_analyse(version: int, num_qubits: int, seq_length: int, data_label: str, random_id: int, learning_rate: float, prediction_step: int, epochs: int, lamb: float, trainable_encoding: bool) -> None:
    """
    Trains and analyzes a quantum model using the provided parameters.

    Args:
        version (int): Version identifier for the training.
        num_qubits (int): Number of qubits in the quantum circuit.
        seq_length (int): Sequence length for the input data.
        data_label (str): Label describing the type of data being processed.
        random_id (int): Random seed for reproducibility.
        learning_rate (float): Learning rate for the optimizer.
        prediction_step (int): Step into the future for prediction.
        epochs (int): Number of training epochs.
        lamb (float): Regularization coefficient.
        trainable_encoding (bool): Whether the encoding weights are trainable.
    """
    model = circuit.Circuit(num_qubits=num_qubits, seq_length=seq_length, data_label=data_label, random_id=random_id, trainable_encoding=trainable_encoding)
    data_handler = DataHandling(data_label=data_label, seq_length=seq_length, prediction_step=prediction_step, random_id=random_id)
    trainer = Trainer(model=model, lamb=lamb, learning_rate=learning_rate, epochs=epochs, print_gradients=False)
    inputs_training, labels_training, inputs_testing, labels_testing = data_handler.get_training_and_test_data()

    cost_training, mse_cost_training, mse_cost_testing, trained_model = trainer.train(inputs_training, labels_training, inputs_testing, labels_testing)

    analyzer = Analyzer(version=version, model=model, trainer=trainer, data_handler=data_handler)
    analyzer.create_directory()
    analyzer.save_training_output(trained_model, cost_training=cost_training, mse_cost_training=mse_cost_training, mse_cost_testing=mse_cost_testing)
    analyzer.plot_cost(cost_training=cost_training, mse_cost_training=mse_cost_training, mse_cost_testing=mse_cost_testing)
    analyzer.evaluate_trained_model(inputs_testing, labels_testing)
    analyzer.evaluate_with_test_noise(inputs_testing, labels_testing)
    analyzer.plot_r_prediction(inputs_training, labels_training, inputs_testing, labels_testing)

if __name__ == '__main__':
    start = time.time()
    def none_or_type(type_):
        def convert(value):
            if value == "None":
                return None
            if type_ == bool:
                return ast.literal_eval(value)
            return type_(value)
        return convert

    parser = argparse.ArgumentParser(description='Train with specific arguments')

    parser.add_argument('-version', '--version', type=none_or_type(int), help='version of the code, in case of change of models, training or data')
    parser.add_argument('-model', '--model_name', type=none_or_type(str), help='model to train e.g. vqc, qlstm_paper, qlstm_linear_enhanced_paper, qrnn_paper, lstm, rnn, mlp')
    parser.add_argument('-data', '--data_label', type=none_or_type(str), help='data e.g. lorenz_1000, mackey_glass_1000_default, henon_map_1000_default')
    parser.add_argument('-id', '--random_id', type=none_or_type(int), help='random id to initialize the weights')
    parser.add_argument('-lr', '--learning_rate', type=none_or_type(float), help='step size of the Adam optimizer')
    parser.add_argument('-num_qubits', '--number_qubits', type=none_or_type(int), help='number of qubits used')
    parser.add_argument('-hidden_size', '--hidden_size', type=none_or_type(int), help='hidden size of the rnn, lstm, qlstm_linear_enhanced_paper, qrnn_paper')
    parser.add_argument('-ansatz', '--type_of_ansatz', type=none_or_type(str), help='type of quantum circuit ansatz')
    parser.add_argument('-seq_length', '--sequence_length', type=none_or_type(int), help='sequence length of data')
    parser.add_argument('-pred_step', '--prediction_step', type=none_or_type(int), help='step into the future on which the models are trained on')
    parser.add_argument('-batch_size', '--batch_size', type=none_or_type(int), help='batch size for training')
    parser.add_argument('-epochs', '--epochs', type=none_or_type(int), help='number of epochs for training')
    parser.add_argument('-lamb', '--lamb', type=none_or_type(float), help='lambda for the loss function')
    parser.add_argument('-trainable_encoding', '--trainable_encoding', type=none_or_type(bool), help='if the encoding is trainable or not')


    args = parser.parse_args()
    if args.version is not None:
        version = args.version
        model_name = args.model_name
        random_id = args.random_id
        learning_rate = args.learning_rate
        ansatz = args.type_of_ansatz
        data_label = args.data_label
        seq_length = args.sequence_length
        prediction_step = args.prediction_step
        num_qubits = args.number_qubits
        hidden_size = args.hidden_size
        batch_size = args.batch_size
        epochs = args.epochs
        lamb = args.lamb
        trainable_encoding = args.trainable_encoding
    else:
        version = 110
        random_id = 0
        learning_rate = 0.01
        data_label = "logistic_map_1000_chaos"
        seq_length = 3
        prediction_step = 1
        num_qubits = 3
        epochs = 10
        lamb = 0.001

        trainable_encoding = True

    train_and_analyse(version=version, num_qubits=num_qubits, seq_length=seq_length, data_label=data_label, random_id=random_id, learning_rate=learning_rate, prediction_step=prediction_step, epochs=epochs, lamb=lamb, trainable_encoding=trainable_encoding)
    
    end = time.time()
    print("total_time for training and analyzing= ", end-start, flush=True)