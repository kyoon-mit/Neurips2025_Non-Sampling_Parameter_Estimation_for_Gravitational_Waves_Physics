import torch
from torch.utils.data import Dataset

class ToyDataGenerator(Dataset):
    def __init__(self, data):
        self.theta_unshifted_vals = data['theta_u'] # unshifted parameters
        self.theta_shifted_vals   = data['theta_s'] # shifted parameters
        self.data_unshifted_vals  = data['data_u']  # unshifted data
        self.data_shifted_vals    = data['data_s']  # shifted data
        self.event_id = data['event_id']

    def __len__(self):
        return self.theta_unshifted_vals.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.theta_unshifted_vals[idx].to(dtype=torch.float32),
            self.theta_shifted_vals[idx].to(dtype=torch.float32),
            self.data_unshifted_vals[idx].to(dtype=torch.float32),
            self.data_shifted_vals[idx].to(dtype=torch.float32),
            self.event_id[idx].to(dtype=torch.int32),
        )