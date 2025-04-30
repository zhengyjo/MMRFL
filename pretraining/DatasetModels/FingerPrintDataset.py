import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

class FingerPrintDataset(Dataset):
    def __init__(self, files, config):
        self.files = files
        self.fingerprint_path = config.fingerprint_path

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        return self.files[idx]

    def __getitem__(self, index):
        file_path = os.path.join(self.fingerprint_path, self.files[index])

        with open(file_path, 'rb') as pickle_file:
            loaded_fingerprint = pickle.load(pickle_file)
        tensor_fingerprint = torch.Tensor(loaded_fingerprint)
        return tensor_fingerprint

    def collate_fn(self, batch):
        # Separate preprocessed NMR data and raw NMR data
        fingerprint = [item for item in batch]

        # Stack the preprocessed NMR data along a new dimension (batch dimension)
        fingerprint_tensor = torch.stack(fingerprint, dim=0)

        return fingerprint_tensor



