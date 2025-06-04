# RobustnessGeneralizationQML
Code for the Book Chapter "The interplay of robustness and generalization in quantum machine learning"


## Installation

To install the required packages, run:

```pip install -r requirements.txt```

## Structure of the repository
- `TimeseriesData` contains the scrips and data for generating the logistic map data
- `utils` contains all scripts relevant for training the quantum model
- `Data` contains the training results
- `Create_Plots` contains the scripts to reproduce the plots from the Chapter
- `Plots` contains the plots of the Chapter

## Running the training and evaluation

To run the training, run:

```python3 ./utils/submit_multiple_jobs.py -slurm "no"```

When changing the flag `-slurm "no"` to `-slurm "yes"` the training can efficiently be parallelized on a slurm cluster. 
