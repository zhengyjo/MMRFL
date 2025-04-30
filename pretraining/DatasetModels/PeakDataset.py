import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class PeakDataset(Dataset):
    def __init__(self, files, config):
        self.files = files
        self.cnmr_path = config.cnmr_path
        self.temperature = config.cnmr_temperature
        self.diff_temperature = config.cnmr_diff_temperature

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        return self.files[idx]

    def __getitem__(self, index):
        file_path = os.path.join(self.cnmr_path, self.files[index])

        df = pd.read_csv(file_path)
        df.sort_values(by='atom', inplace=True)
        peak = df['ppm'].values.tolist()
        peak_tensor = torch.tensor(peak, dtype=torch.float32)
        return peak_tensor

    def collate_fn(self, batch):
        ppm = [item for item in batch]
        ppm = torch.cat(ppm, dim=0)
        ppm = ppm.view(-1, 1)
        ppm_diff = torch.abs(ppm - ppm.t())
        ppm_diff += self.temperature
        ppm_diff = 1 / ppm_diff * self.diff_temperature

        return ppm_diff




