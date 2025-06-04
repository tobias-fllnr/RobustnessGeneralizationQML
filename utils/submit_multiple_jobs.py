import itertools
import os
import json
import argparse


def generate_combinations(param_dict: dict) -> list:
    """
    Generates all possible combinations of parameters from a dictionary.

    Args:
        param_dict (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        list: List of dictionaries, each representing a combination of parameters.
    """
    keys, values = zip(*param_dict.items())
    combinations: list = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def submit_job(job_name: str, memory: int, command: str, script_filename: str, combo: dict) -> None:
    """
    Submits a job to SLURM using a generated script.

    Args:
        job_name (str): Name of the SLURM job.
        memory (int): Memory allocation for the job in GB.
        command (str): Command to execute in the SLURM job.
        script_filename (str): Filename for the SLURM script.
        combo (dict): Dictionary of parameter combinations for the job.
    """
    slurm_template: str = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}GB
#SBATCH --time=02:00:00
#SBATCH --mail-user=tfellner@icp.uni-stuttgart.de
#SBATCH --mail-type=FAIL
#SBATCH --error={job_name}_%j.err

{command}
"""
    slurm_script: str = slurm_template.format(job_name=job_name, memory=memory, command=command)

    with open(script_filename, "w") as script_file:
        script_file.write(slurm_script)

    os.system(f"sbatch {script_filename}")
    print(f"Job submitted for combination: {combo}")
    os.remove(script_filename)


def determine_memory(num_qubits: int) -> int:
    """
    Determines the memory allocation based on the number of qubits.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.

    Returns:
        int: Memory allocation in GB.

    Raises:
        ValueError: If the number of qubits exceeds the supported range.
    """
    if num_qubits <= 4:
        memory: int = 4
    elif num_qubits <= 8:
        memory: int = 8
    elif num_qubits <= 10:
        memory: int = 16
    elif num_qubits <= 16:
        memory: int = 64
    else:
        raise ValueError("Number of qubits exceeds the supported range for memory allocation.")
    return memory


def run_experiment(config: dict, slurm: bool) -> None:
    """
    Runs an experiment either on SLURM or locally.

    Args:
        config (dict): Dictionary containing experiment configuration parameters.
        slurm (bool): Whether to submit the job to SLURM or run locally.
    """
    # Extract parameters
    version: str = config["version"]
    random_id: int = config["random_ids"]
    data_label: str = config["data_labels"]
    learning_rate: float = config["learning_rates"]
    num_qubits: int = config["num_qubits"]
    sequence_length: int = config["sequence_lengths"]
    prediction_step: int = config["prediction_steps"]
    epoch: int = config["epochs"]
    lamb: float = config["lambs"]
    trainable_encoding: bool = config["trainable_encoding"]

    if slurm:
        job_name: str = f"{version}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{sequence_length}_{prediction_step}_{epoch}_{lamb}_{trainable_encoding}"
        command: str = f"srun python3 ./utils/training_and_analyzing.py -version {version} -data {data_label} -id {random_id} -lr {learning_rate} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -epochs {epoch} -lamb {lamb} -trainable_encoding {trainable_encoding}"
        script_filename: str = f"submit_{version}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{prediction_step}_{sequence_length}_{epoch}_{lamb}_{trainable_encoding}.sh"
        memory: int = determine_memory(num_qubits)
        submit_job(job_name, memory, command, script_filename, config)
    else:
        # Run the command directly without SLURM
        command: str = f"python3 ./utils/training_and_analyzing.py -version {version} -data {data_label} -id {random_id} -lr {learning_rate} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -epochs {epoch} -lamb {lamb} -trainable_encoding {trainable_encoding}"
        os.system(command)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Submission Controller")
    parser.add_argument(
        "-slurm", "--slurm", type=str, help="yes or no"
    )

    args: argparse.Namespace = parser.parse_args()
    slurm: bool = args.slurm == "yes"

    try:
        # Read the JSON file
        with open("./configurations_totrain.json", 'r') as file:
            data: list = json.load(file)

        if not isinstance(data, list):
            print("The JSON file does not contain a list of dictionaries.")

        # Process each element in the list
        for item in data[:]:  # Iterate over a copy to allow modification
            try:
                # Load the item as a dictionary
                if isinstance(item, dict):
                    print("Processing:", item)
                    # Generate combinations and run experiments
                    combinations: list = generate_combinations(item)
                    for combo in combinations:
                        run_experiment(combo, slurm)
                else:
                    raise ValueError("Item is not a dictionary")
            except Exception as e:
                print(f"Error processing item: {item}. Error: {e}")

    except FileNotFoundError:
        print(f"The file {'configurations_totrain.json'} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in the file {'configurations_totrain.json'}.")

    print("All jobs submitted.")