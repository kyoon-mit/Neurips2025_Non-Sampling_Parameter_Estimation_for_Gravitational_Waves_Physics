<h2 style="color: orange"> Setup </h2>

### Prerequisities
- Conda
- Slurm submission

### Setup script
In order to setup the environment correctly, first open a bash terminal and run
```
source install.sh
```
This will set the necessary environment variables as well as generate the correct ```env.yaml``` file.

### Create Conda environment
Create the conda environment from the ```.yaml``` file by running,
```
conda env create -f env.yaml
```
Then, enter the conda environment.
```
conda activate env
```
Obviously, you are free to choose whatever name you want for the conda environment.

<h2 style="color: orange"> Reproduce all paper results </h2>

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