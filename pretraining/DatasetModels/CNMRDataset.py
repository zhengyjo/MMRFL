import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class CNMRDataset(Dataset):
    def __init__(self, files, config):
        self.files = files
        self.cnmr_path = config.cnmr_path
        self.device = config.device

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        return self.files[idx]

    def __getitem__(self, index):
        file_path = self.cnmr_path + self.files[index]

        # Load NMR data from file with optimized dtype specification
        df = pd.read_csv(file_path, dtype={'atom': int, 'ppm': float})
        df.sort_values(by='atom', inplace=True)

        # Extract NMR data and apply the preprocessing transform
        nmr = df['ppm'].values.tolist()
        nmr_tensor = self.preprocess_nmr(nmr)
        return nmr_tensor.to(self.device)

    def collate_fn(self, batch):
        # Separate preprocessed NMR data and raw NMR data
        nmr_tensor = [item for item in batch]

        # Stack the preprocessed NMR data along a new dimension (batch dimension)
        nmr_tensor = torch.stack(nmr_tensor, dim=0)
 
        return nmr_tensor

    def preprocess_nmr(self, nmr, scale=10, min_value=-50, max_value=350):
        units = (max_value - min_value) * scale
        item = np.zeros(units)
        nmr = [round((value - min_value) * scale) for value in nmr]

        for index, value in enumerate(nmr):
            if value < 0:
                item[0] = 1
            elif value >= units:
                item[-1] = 1
            else:
                item[value] = 1

        item = torch.from_numpy(item).to(torch.float32)
        return item




