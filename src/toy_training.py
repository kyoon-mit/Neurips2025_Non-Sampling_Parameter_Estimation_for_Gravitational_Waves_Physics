import os
import lightning as L
import torch
from torch import Dataset, DataLoader

class ToyDataModule(L.LightningDataModule):
    def __init__(self,
        data_type, # 'DHO' or 'SineGaussian'
        data_dir, # Path to the directory where the dataset is saved.
        noise_profile, # Valid inputs: 'gaussian_smear'
        noise_strength, # For Gaussian, this is sigma.
        **kwargs
    ):
        super().__init__()
        file_name_format = f'{data_type}_{noise_profile}_{noise_strength:.1f}.pt'
        self.test_data_path = os.path.join(data_dir, data_type, f'test_{file_name_format}')
        self.train_data_path = os.path.join(data_dir, data_type, f'train_{file_name_format}')
        self.val_data_path = os.path.join(data_dir, data_type, f'val_{file_name_format}')        

    def prepare_data(self):
        # Search whether data already exists
        # If not, create data.
        if not (os.path.exists(self.test_data_path) and
            os.path.exists(self.train_data_path) and
            os.path.exists(self.val_data_path)
        ):
            import neurips2025.src.utils.toy_functions as toy_functions
            t_vals = torch.linspace(t_start, t_end, num_points).repeat(num_samples, 1)

            if data_type == 'DHO':
                y_vals = toy_functions.func_dho(t_vals, omega_0, beta, shift, method='torch')
            elif data_type == 'SineGaussian':
                y_vals = toy_functions.func_sg(t_vals, f_0, tau, shift, method='torch')
            match noise_profile:
                case 'gaussian_smear':
                    y_vals = toy_functions.add_noise_gaussian_smearing(t_vals, y_vals, method='torch')
                case _:
                    pass
            # Save data

            indices = torch.randperm(num_samples)
        
        return

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = \
                Subset(self.dataset, self.indices['train_indices']), \
                Subset(self.dataset, self.indices['val_indices'])

        # Assign test dataset for use in dataloader
        elif stage in ("test", "predict"):
            self.test_dataset = Subset(self.dataset, self.indices['test_indices'])

        if stage == 'fit':
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
    

class ToyDataGenerator(Dataset):
    def __init__(self, data):
        self.theta_unshifted_vals = data['theta_u']
        self.theta_shifted_vals   = data['theta_s']
        self.data_unshifted_vals  = data['data_u']
        self.data_shifted_vals    = data['data_s']
        self.event_id = data['event_id']

    def __len__(self):
        return self.theta_unshifted_vals.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return shifted and unshifted theta and data
        return (
            self.theta_unshifted_vals[idx].to(dtype=torch.float32),
            self.theta_shifted_vals[idx].to(dtype=torch.float32),
            self.data_unshifted_vals[idx].to(dtype=torch.float32),
            self.data_shifted_vals[idx].to(dtype=torch.float32),
            self.event_id[idx].to(dtype=torch.int32),
        )