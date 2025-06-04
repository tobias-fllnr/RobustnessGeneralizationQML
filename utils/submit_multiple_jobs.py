import itertools
import os
import json
import argparse


def generate_combinations(param_dict):
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def submit_job(job_name, memory, command, script_filename, combo):
    slurm_template = """#!/bin/bash
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
    slurm_script = slurm_template.format(job_name=job_name, memory=memory, command=command)

    with open(script_filename, "w") as script_file:
        script_file.write(slurm_script)

    os.system(f"sbatch {script_filename}")
    print(f"Job submitted for combination: {combo}")
    os.remove(script_filename)


def determine_memory(num_qubits):
    if num_qubits <= 4:
        memory = 4
    elif num_qubits <= 8:
        memory = 8
    elif num_qubits <= 10:
        memory = 16
    elif num_qubits <= 16:
        memory = 64
    else:
        raise ValueError("Number of qubits exceeds the supported range for memory allocation.")
    return memory


def run_experiment(config, slurm):
    # Extract parameters
    version = config["version"]
    random_id = config["random_ids"]
    data_label = config["data_labels"]
    learning_rate = config["learning_rates"]
    num_qubits = config["num_qubits"]
    sequence_length = config["sequence_lengths"]
    prediction_step = config["prediction_steps"]
    epoch = config["epochs"]
    lamb = config["lambs"]
    trainable_encoding = config["trainable_encoding"]

    if slurm:

        job_name = f"{version}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{sequence_length}_{prediction_step}_{epoch}_{lamb}_{trainable_encoding}"
        command = f"srun python3 ./utils/training_and_analyzing.py -version {version} -data {data_label} -id {random_id} -lr {learning_rate} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -epochs {epoch} -lamb {lamb} -trainable_encoding {trainable_encoding}"
        script_filename = f"submit_{version}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{prediction_step}_{sequence_length}_{epoch}_{lamb}_{trainable_encoding}.sh"
        memory = determine_memory(num_qubits)
        submit_job(job_name, memory, command, script_filename, config)
    
    else:
                # Run the command directly without SLURM
        command = f"python3 ./utils/training_and_analyzing.py -version {version} -data {data_label} -id {random_id} -lr {learning_rate} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -epochs {epoch} -lamb {lamb} -trainable_encoding {trainable_encoding}"
        os.system(command)




def save_training_configuration(item):
    version = item["version"][0]
    # Define the directory path
    path = f"./Submitted_Configurations/Version_{version}"
    
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Find the next available file name
    i = 1
    while True:
        filename = os.path.join(path, f"{i}.json")  # Full path to the file
        if not os.path.exists(filename):  # Check if the file exists
            # Save the JSON file
            with open(filename, 'w') as output_file:
                json.dump(item, output_file, indent=4)
            break
        i += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submission Controller")
    parser.add_argument(
        "-slurm", "--slurm", type=str, help="yes or no")
    
    args = parser.parse_args()
    slurm = args.slurm
    if slurm == "yes":
        slurm = True
    else:
        slurm = False

    try:
        # Read the JSON file
        with open("./configurations_totrain.json", 'r') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            print("The JSON file does not contain a list of dictionaries.")
        
        # Process each element in the list
        for item in data[:]:  # Iterate over a copy to allow modification
            try:
                # Load the item as a dictionary
                if isinstance(item, dict):
                    print("Processing:", item)
                    # Perform your processing logic here
                    
                    combinations = generate_combinations(item)
                    for combo in combinations:
                        run_experiment(combo, slurm)
                    
                    # Save the training configuration to ./Submitted_Configurations
                    save_training_configuration(item)


                else:
                    raise ValueError("Item is not a dictionary")
            except Exception as e:
                print(f"Error processing item: {item}. Error: {e}")
    
    except FileNotFoundError:
        print(f"The file {'configurations_totrain.json'} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in the file {'configurations_totrain.json'}.")

    print("All jobs submitted.")