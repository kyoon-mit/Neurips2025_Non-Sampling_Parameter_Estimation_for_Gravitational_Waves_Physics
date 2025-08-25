# Functional forms for the toy models
# Adapted from these notebooks:
# https://github.com/ML4GW/summer-projects-2023/blob/main/symmetry-informed-flows/notebooks/DampedHarmonicOscillator/sho-baseline-training.ipynb
# https://github.com/ML4GW/summer-projects-2023/blob/main/symmetry-informed-flows/notebooks/SineGaussian/sine-gaussian-baseline-training.ipynb 

import numpy as np
import torch

def func_dho(t, omega_0, beta, shift, method='torch'):
    """Generate damped harmonic oscillator data.

    Args:
        t (float | torch.Tensor | np.ndarray): Time.
        omega_0 (float | torch.Tensor | np.ndarray): Natural frequency.
        beta (float | torch.Tensor | np.ndarray): Damping ratio (0 < beta < 1).
        shift (float | torch.Tensor | np.ndarray): Signal start time.
        method (str): 'torch' or 'numpy'.

    Returns:
        Damped oscillator data (torch.Tensor or np.ndarray).
    """
     # Check the value of beta
    if (beta >= 1) or (beta <= 0):
        raise ValueError('Beta must be between 0 and 1 to be underdamped.')
    # Define functional methods
    if method == 'torch': _sqrt, _exp, _cos = torch.sqrt, torch.exp, torch.cos
    elif method == 'numpy': _sqrt, _exp, _cos = np.sqrt, np.exp, np.cos
    else: raise ValueError(f'Unknown {method=}.')
    osc = _sqrt(1 - beta**2) * omega_0
    t_shifted = t - shift
    data = _exp(- beta * omega_0 * t_shifted) * _cos(osc * t_shifted)
    data[t_shifted < 0] = 0 # Apply 0 when t < shift
    return data

def func_sg(t, f_0, tau, shift, method='torch'):
    """Generate sine-gaussian pulse data.

    Args:
        t (float | torch.Tensor | np.ndarray): Time.
        f_0 (float | torch.Tensor | np.ndarray): Central frequency.
        tau (float | torch.Tensor | np.ndarray): Pulse width.
        shift (float | torch.Tensor | np.ndarray): Signal start time.
        method (str): 'torch' or 'numpy'.

    Returns:
        Sine-Gaussian data (torch.Tensor or np.ndarray).
    """
    t_shifted = t - shift
    if method == 'torch': _exp, _sin, _pi = torch.exp, torch.sin, torch.pi
    elif method == 'numpy': _exp, _sin, _pi = np.exp, np.sin, np.pi
    else: raise ValueError(f'Unknown {method=}.')
    data = _exp(- t_shifted**2 / (tau**2)) * _sin(2 * _pi * f_0 * t_shifted)
    return data

### Noise in time-domain ###
def add_gaussian_noise_time_domain(y_vals, sigma, method='torch'):
    """Add Gaussian noise in the time domain.

    Adds noise drawn from N(0, sigma) to each component of y(t).

    Args:
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        sigma (float): Standard deviation of the Gaussian noise.
        method (str): 'torch' or 'numpy'.

    Returns:
        Gaussian-smeared data (torch.Tensor or np.ndarray).
    """
    if method=='torch':
        y_noise = sigma * torch.randn(size=y_vals.size()).to(dtype=torch.float32)
    elif method=='numpy':
        y_noise = sigma * np.random.randn(*y_vals.shape).astype(np.float32)
    else: raise ValueError(f'Unknown {method=}.')
    y_vals += y_noise
    return y_vals

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
    # Sanity check
    print(y_freq)
    return y_freq

def add_gaussian_white_noise(y_vals, sigma, method='torch'):
    """Add Gaussian white noise in the frequency domain.

    Converts time-domain data y(t) to the frequency domain,
    adds white noise drawn from N(0, sigma) to each frequency component,
    and transforms the result back to the time domain.

    Args:
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        sigma (float): Standard deviation of the Gaussian noise.
        method (str): 'torch' or 'numpy'.

    Returns:
        Data with added Gaussian white noise (torch.Tensor or np.ndarray).
    """
    if method=='torch':
        y_freq = torch.fft.rfft(y_vals, dim=-1)
        y_noise = sigma * torch.randn(size=y_freq.size()).to(dtype=torch.float32)
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = torch.fft.irfft(y_freq + y_noise)
    elif method=='numpy':
        y_freq = np.fft.rfft(y_vals, axis=-1)
        y_noise = sigma * np.random.randn(*y_freq.shape).astype(np.float32)
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = np.fft.irfft(y_freq + y_noise)
    else: raise ValueError(f'Unknown {method=}.')
    return y_vals

def add_gaussian_pink_noise(t_start, t_end, t_num_points, y_vals, sigma, method='torch'):
    """Add Gaussian pink noise in the frequency domain.

    Converts time-domain data y(t) to the frequency domain,
    adds pink noise drawn from N(0, sigma) to each frequency component,
    and transforms the result back to the time domain.

    Args:
        t_start (float): Starting value of the time domain.
        t_end (float): Ending value of the time domain.
        t_num_points (int): Number of discrete points in the time domain.
        y_vals (torch.Tensor or np.ndarray): Amplitude values in the time domain.
        sigma (float): Standard deviation of the Gaussian noise
        method (str): 'torch' or 'numpy'.

    Returns:
        Data with added Gaussian pink noise (torch.Tensor or np.ndarray).
    """
    if method=='torch':
        y_freq = torch.fft.rfft(y_vals, dim=-1)
        x_freq = torch.fft.rfftfreq(n=t_num_points, d=(t_end-t_start)/t_num_points)
        y_noise = sigma * torch.randn(size=y_freq.size()).to(dtype=torch.float32)
        y_noise = y_noise/x_freq
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = torch.fft.irfft(y_freq + y_noise)
    elif method=='numpy':
        y_freq = np.fft.rfft(y_vals, axis=-1)
        y_noise = sigma * np.random.randn(*y_freq.shape).astype(np.float32)
        y_noise = y_noise/x_freq
        y_noise = _zero_first_frequency(y_noise) # Always set the zeroeth frequency amplitude to zero [[0, ...], [0, ...], ...]
        y_vals = np.fft.irfft(y_freq + y_noise)
    else: raise ValueError(f'Unknown {method=}.')
    return y_vals