from copy import deepcopy
import torch
import numpy as np

def load_priors(priors_cfg: dict, num_samples: int, seed=42, method='torch', **kwargs):
    priors = {}
    for name, cfg in priors_cfg.items():
        if cfg['distribution'] == 'uniform':
            low, high = cfg['low'], cfg['high']
            if method=='torch':
                rng = torch.Generator().manual_seed(seed)
                priors[name] = low + (high - low) * torch.rand(num_samples, 1)
            elif method=='numpy':
                rng = np.random.default_rng(seed)
                priors[name] = rng.uniform(low, high, size=(num_samples, 1))
            else: raise ValueError(f'Unknown {method=}.')
        else:
            raise ValueError(f'Unknown distribution {cfg["distribution"]}')
    return priors

def load_unshifted_priors(priors: dict, method='torch'):
    priors_unshifted = deepcopy(priors)
    shape = priors_unshifted['shift'].shape
    if method=='torch': priors_unshifted['shift'] = torch.ones(shape)
    elif method=='numpy': priors_unshifted['shift'] = np.ones(shape)
    else:  raise ValueError(f'Unknown {method=}.')
    return priors_unshifted