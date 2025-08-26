import os, yaml
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
        priors_shifted = load_priors(config['priors'], **config)
        priors_unshifted = load_unshifted_priors(priors_shifted, method=config['method'])
        datatype = config['datatype']
        noisetype = config['noisetype']
        match datatype:
            case 'dho': _signal = func_dho
            case 'sg': _signal = func_sg
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
        y_vals_shifted = y_vals_shifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)
        y_vals_unshifted = y_vals_unshifted.unsqueeze(1).repeat(1, config['num_repeats'], 1)
    return y_noise, y_vals_shifted, y_vals_unshifted, priors_shifted, priors_unshifted

def plot_toy_data(config_file, y_vals, y_noise, shift):
    import torch
    import matplotlib.pyplot as plt
    with open(config_file) as stream:
        config = yaml.full_load(stream)
        t_vals = torch.linspace(config['t_start'], config['t_end'], config['t_num_points'])
        plt.plot(t_vals, y_vals - y_noise)
        plt.scatter(t_vals, y_noise)
        plt.savefig(f'test_{config["datatype"]}_{config["noisetype"]}_{shift}.png')
        plt.close()
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
    for config_file in config_file_list:
        fp = os.path.join(config_dir, config_file)
        y_noise, y_vals_shifted, y_vals_unshifted, priors_shifted, priors_unshifted = generate_toy_data(config_file=fp)
        plot_toy_data(fp, y_vals_shifted[0][0], y_noise[0], shift='shifted')
        plot_toy_data(fp, y_vals_unshifted[0][0], y_noise[0], shift='unshifted')