import os, yaml
from functions.toy_functions import *
from datasets.load_priors import load_priors
from datasets.data_structures import ToyDataGenerator

def generate_toy_data(config_file):
    """Generate toy data.

    Args:
        config_file (str): Path to the config .yaml file.

    Returns:
    
    """
    with open(config_file) as stream:
        config = yaml.full_load(stream)
        num_samples = config['num_samples']
        priors = load_priors(config['priors'], num_samples=num_samples)
        datatype = config['datatype']
        noisetype = config['noisetype']
        match datatype:
            case 'dho': _signal = func_dho
            case 'sg': _signal = func_sg
            case _: raise ValueError(f'Unknown {datatype=}.')
        y_vals = _signal(**config, **priors)
        match noisetype:
            case 'gaussian_white': _noise = add_gaussian_white_noise
            case 'gaussian_smear': _noise = add_gaussian_noise_time_domain
            case 'gaussian_pink': _noise = add_gaussian_pink_noise
            case _: raise ValueError(f'Unknown {noisetype=}.')
        y_vals = _noise(y_vals=y_vals, **config)
        # Create a tensor or array of shape (num_samples, num_repeats, t_num_points)
        y_vals = y_vals.unsqueeze(1).repeat(1, config['num_repeats'], 1)
    return y_vals