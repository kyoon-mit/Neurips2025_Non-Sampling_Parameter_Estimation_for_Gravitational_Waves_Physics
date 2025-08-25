## Setup

### Prerequisities
- Conda
- Slurm submission

### Create Conda environment
Create the conda environment from the ```.yaml``` file in your prefered conda env directory.
```
conda env create -f myenv.yaml --prefix /path/to/your/conda/envs/myenv
```
Then, enter the conda environment.
```
conda activate myenv
```
Obviously, you are free to choose whatever name you want for the conda environment.

### Pip install current project
To ensure that the imports work, go to the top directory and run the following command.
```
pip install -e .
```
This must be done within the conda environment.

### Run environment variables script
There are environment variables which are user-dependent, and must be set correctly. To do so, run the following every time you login.
```
source env.sh
```

## Reproduce all paper results

First, make sure you have followed the instructions in the **Setup**. 

### Section 2: Toy Experiments
Run
```
python reproduce/section2_toy_experiments.py
```
This will
1. Generate the toy datasets.
2. Run the training of the SSM models and the baseline models on the toy datasets.
3. Plot the results.

### Section 3: BNS


Additional Information
--------
## Preparing Toy Dataset
There are two toy datasets in this paper: damped harmonic oscillator and sine-gaussian pulse. In order to generate the toy datasets, simply run
```
python 