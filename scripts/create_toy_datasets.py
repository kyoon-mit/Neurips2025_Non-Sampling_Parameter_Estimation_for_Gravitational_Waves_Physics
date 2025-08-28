import os, yaml
import torch
import numpy as np
from functions.toy_functions import *
from datasets.load_priors import load_priors, load_unshifted_priors
from datasets.data_structures import ToyDataGenerator

def generate_toy_data(config_file):
    """Generate toy data.

    Args:
        config_file (str): Path to the config .yaml file.

    Returns:
        y_noise (torch.Tensor or np.ndarray): The added noise.
        y_vals_shifted (torch.Tensor or np.ndarray): The shifted signal values.
        y_vals_unshifted (torch.Tensor or np.ndarray): The unshifted signal values.
        priors_shifted (dict): The shifted priors.
        priors_unshifted (dict): The unshifted priors.
    """
    with open(config_file) as stream:
        config = yaml.full_load(stream)
        method = config['method']
        if method=='torch': event_id = torch.tensor(range(0, config['num_samples']), dtype=torch.int32)
        else: event_id = np.array(range(0, config['num_samples']), dtype=np.int32)
        priors_shifted = load_priors(config['priors'], **config)
        priors_unshifted = load_unshifted_priors(priors_shifted, method=method)
        datatype = config['datatype']
        noisetype = config['noisetype']
        match datatype:
            case 'dho':
                _signal = func_dho
                theta_shifted = _stack_priors_dho(priors_shifted, method=method)
                theta_unshifted = _stack_priors_dho(priors_unshifted, method=method)
            case 'sg':
                _signal = func_sg
                theta_shifted = _stack_priors_sg(priors_shifted, method=method)
                theta_unshifted = _stack_priors_sg(priors_unshifted, method=method)
            case _: raise ValueError(f'Unknown {datatype=}')
        y_vals_shifted = _signal(**config, **priors_shifted)
        y_vals_unshifted = _signal(**config, **priors_unshifted)
        match noisetype:
            case 'gaussian_white': _noise = add_gaussian_white_noise
            case 'gaussian_smear': _noise = add_gaussian_noise_time_domain
            case 'gaussian_pink': _noise = add_gaussian_pink_noise
            case _: raise ValueError(f'Unknown {noisetype=}')
        y_vals_shifted, y_noise = _noise(y_vals=y_vals_shifted, **config)
        y_vals_unshifted = y_vals_unshifted + y_noise
        
        # Create a tensor or array of shape (num_samples, num_repeats, t_num_points)
        theta_shifted = theta_shifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)
        theta_unshifted = theta_unshifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)
        y_vals_shifted = y_vals_shifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)
        y_vals_unshifted = y_vals_unshifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)

        # split indices
        num_total  = len(event_id)
        train_size = int(0.8 * num_total)
        val_size   = int(0.1 * num_total)

        indices   = np.random.permutation(event_id)
        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        # Save the data
        datatype = config['datatype']
        noisetype = config['noisetype']
        sigma = config['sigma']
        save_dir = os.path.join(os.environ.get('PROJECT_DIR'), 'data', datatype.upper())
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            save_path = os.path.join(save_dir, f'{split}_{datatype}_{noisetype}_sigma{sigma}.pt')
            _save_split(save_path, idx, y_vals_shifted, y_vals_unshifted, theta_shifted, theta_unshifted, event_id)
    return

# Helper to create theta stack
def _stack_priors_dho(priors, method='torch'):
    if method=='torch': theta = torch.stack([priors['omega_0'], priors['beta'], priors['shift']], dim=-1)
    elif method=='numpy': theta = np.stack([priors['omega_0'], priors['beta'], priors['shift']], axis=-1)
    else: raise ValueError(f'Unknown {method=}.')
    return theta

def _stack_priors_sg(priors, method='torch'):
    if method=='torch': theta = torch.stack([priors['f_0'], priors['tau'], priors['shift']], dim=-1)
    elif method=='numpy': theta = np.stack([priors['f_0'], priors['tau'], priors['shift']], axis=-1)
    else: raise ValueError(f'Unknown {method=}.')
    return theta

# Helper to slice and save each split
def _save_split(save_path, idx, y_vals_shifted, y_vals_unshifted, theta_shifted, theta_unshifted, event_id):
    torch.save({
        'theta_u': theta_unshifted[idx],
        'theta_s': theta_shifted[idx],
        'data_u':  y_vals_unshifted[idx],
        'data_s':  y_vals_shifted[idx],
        'event_id': event_id[idx],
    },
    save_path,
    _use_new_zipfile_serialization=True)
    print(f'Saved data to {save_path}')
    return

if __name__=='__main__':
    config_file_list = [
        'dho_dataset_gaussian_smear.yaml',
        'dho_dataset_gaussian_white.yaml',
        'dho_dataset_gaussian_pink.yaml',
        'sg_dataset_gaussian_smear.yaml',
        'sg_dataset_gaussian_white.yaml',
        'sg_dataset_gaussian_pink.yaml'
    ]
    project_dir = os.environ.get('PROJECT_DIR')
    config_dir = os.path.join(project_dir, 'configs')
    for fname in config_file_list:
        config_file = os.path.join(config_dir, fname)
        generate_toy_data(config_file)