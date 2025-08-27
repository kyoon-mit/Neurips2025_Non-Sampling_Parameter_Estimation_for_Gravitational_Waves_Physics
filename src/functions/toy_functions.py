# Functional forms for the toy models
# Partially adapted from these notebooks:
# https://github.com/ML4GW/summer-projects-2023/blob/main/symmetry-informed-flows/notebooks/DampedHarmonicOscillator/sho-baseline-training.ipynb
# https://github.com/ML4GW/summer-projects-2023/blob/main/symmetry-informed-flows/notebooks/SineGaussian/sine-gaussian-baseline-training.ipynb 

import numpy as np
import torch
import colorednoise as cn

def func_dho(t_start, t_end, t_num_points, omega_0, beta, shift, method='torch', **kwargs):
    """Generate damped harmonic oscillator data.

    Args:
        t_start (float): Starting value of time.
        t_end (float): Ending value of time.
        t_num_points (int): Number of data points.
        omega_0 (float | torch.Tensor | np.ndarray): Natural frequency.
        beta (float | torch.Tensor | np.ndarray): Damping ratio (0 < beta < 1).
        shift (float | torch.Tensor | np.ndarray): Signal start time.
        method (str): 'torch' or 'numpy'.
        **kwargs

    Returns:
        Damped oscillator data (torch.Tensor or np.ndarray).
    """
    # Define functional methods
    if method == 'torch':
        _sqrt, _exp, _cos, _linspace = torch.sqrt, torch.exp, torch.cos, torch.linspace
    elif method == 'numpy':
        _sqrt, _exp, _cos, _linspace = np.sqrt, np.exp, np.cos, np.linspace
    else: raise ValueError(f'Unknown {method=}.')
    t = _linspace(start=t_start, end=t_end, steps=t_num_points)
    osc = _sqrt(1 - beta**2) * omega_0
    t_shifted = t - shift
    data = _exp(- beta * omega_0 * t_shifted) * _cos(osc * t_shifted)
    data[t_shifted < 0] = 0 # Apply 0 when t < shift
    return data

def func_sg(t_start, t_end, t_num_points, f_0, tau, shift, method='torch', **kwargs):
    """Generate sine-gaussian pulse data.

    Args:
        t_start (float): Starting value of time.
        t_end (float): Ending value of time.
        t_num_points (int): Number of data points.
        f_0 (float | torch.Tensor | np.ndarray): Central frequency.
        tau (float | torch.Tensor | np.ndarray): Pulse width.
        shift (float | torch.Tensor | np.ndarray): Signal start time.
        method (str): 'torch' or 'numpy'.
        **kwargs

    Returns:
        Sine-Gaussian data (torch.Tensor or np.ndarray).
    """
    if method == 'torch':
        _exp, _sin, _pi, _linspace = torch.exp, torch.sin, torch.pi, torch.linspace
    elif method == 'numpy':
        _exp, _sin, _pi, _linspace = np.exp, np.sin, np.pi, np.linspace
    else: raise ValueError(f'Unknown {method=}.')
    t = _linspace(start=t_start, end=t_end, steps=t_num_points)
    t_shifted = t - shift
    data = _exp(- t_shifted**2 / (tau**2)) * _sin(2 * _pi * f_0 * t_shifted)
    return data

### Noise in time-domain ###
def add_gaussian_noise_time_domain(y_vals, sigma, method='torch', seed=42, **kwargs):
    """Add Gaussian noise in the time domain.

    Adds noise drawn from N(0, sigma) to each component of y(t).

    Args:
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        sigma (float): Standard deviation of the Gaussian distribution.
        method (str): 'torch' or 'numpy'. If omitted, defaults to 'torch'.
        seed (int): Random seed. Defaults to 42.
        **kwargs

    Returns:
        Gaussian-smeared data (torch.Tensor or np.ndarray).
        noise (torch.Tensor or np.ndarray): The added noise.
    """
    if method=='torch':
        rng = torch.Generator().manual_seed(seed)
        y_noise = sigma * torch.randn(size=y_vals.size(), generator=rng).to(dtype=torch.float32)
    elif method=='numpy':
        rng = np.random.default_rng(seed)
        y_noise = sigma * rng.standard_normal(size=y_vals.shape).astype(np.float32)
    else: raise ValueError(f'Unknown {method=}.')
    y_vals += y_noise
    return y_vals, y_noise

### Noise in frequency-domain ###
def _zero_first_frequency(y_freq):
    """Set the first frequency component along the last axis to zero.

    This function works for arrays or tensors of any dimension:
      - 1D: sets y_freq[0] = 0
      - 2D: sets y_freq[:, 0] = 0
      - 3D: sets y_freq[:, :, 0] = 0
      - and so on for higher dimensions.

    Args:
        y_freq (torch.Tensor or np.ndarray): Frequency-domain amplitudes.

    Returns:
        torch.Tensor or np.ndarray: The same array/tensor with the
        first frequency component set to zero.
    """
    y_freq[(..., 0)] = 0
    return y_freq

def add_gaussian_white_noise(y_vals, sigma, method='torch', seed=42, **kwargs):
    """Add Gaussian white noise in the frequency domain.

    Converts time-domain data y(t) to the frequency domain,
    adds white noise drawn from N(0, sigma) to each frequency component,
    and transforms the result back to the time domain.

    Args:
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        sigma (float): Standard deviation of the Gaussian distribution.
        method (str): 'torch' or 'numpy'. If omitted, defaults to 'torch'.
        seed (int): Random seed. Defaults to 42.
        **kwargs

    Returns:
        Data with added Gaussian white noise (torch.Tensor or np.ndarray).
        noise (torch.Tensor or np.ndarray): The added noise in time domain.
    """
    if method=='torch':
        y_freq = torch.fft.rfft(y_vals, dim=-1)
        rng = torch.Generator().manual_seed(seed)
        y_noise = sigma * torch.randn(size=y_freq.size(), generator=rng).to(dtype=torch.float32)
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = torch.fft.irfft(y_freq + y_noise)
    elif method=='numpy':
        y_freq = np.fft.rfft(y_vals, axis=-1)
        rng = np.random.default_rng(seed)
        y_noise = sigma * rng.standard_normal(*y_freq.shape).astype(np.float32)
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = np.fft.irfft(y_freq + y_noise)
    else: raise ValueError(f'Unknown {method=}.')
    return y_vals, torch.fft.irfft(y_noise)

def add_gaussian_pink_noise(t_start, t_end, t_num_points, y_vals, sigma, method='torch', seed=42, **kwargs):
    """Add pink noise in the frequency domain.

    For description of the algorithm, refer to
    Timmer, J. and Koenig, M.: On generating power law noise. Astron. Astrophys. 300, 707-710 (1995).

    The noise is generated via the package,
    https://github.com/felixpatzelt/colorednoise.

    Args:
        t_start (float): Starting value of the time domain.
        t_end (float): Ending value of the time domain.
        t_num_points (int): Number of discrete points in the time domain.
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        method (str): 'torch' or 'numpy'. If omitted, defaults to 'torch'.
        seed (int): Random seed. Defaults to 42.
        **kwargs

    Returns:
        Data with added Gaussian pink noise (torch.Tensor or np.ndarray).
        noise (torch.Tensor or np.ndarray): The added noise in time domain.
    """
    y_noise = sigma * cn.powerlaw_psd_gaussian(exponent=1, size=y_vals.shape, random_state=seed)
    if method=='torch': y_noise = torch.tensor(y_noise, dtype=torch.float32)
    elif method=='numpy': y_noise.astype(np.float32)
    else: raise ValueError(f'Unknown {method=}.')
    return y_vals, y_noise