import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class PeakSignDataset(Dataset):
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
        tensor = ppm.clone()
        
        # Get the number of elements in the tensor
        n = tensor.size(0)

        # Create an empty matrix to store the results
        output_matrix = torch.zeros((n, n), dtype=torch.long)

        # Compare each element with every other element
        for i in range(n):
            # Extract the current element to compare against
            current_element = tensor[i]

            # Fill the row of the matrix with the correct comparison values
            output_matrix[i] = torch.where(tensor < current_element,
                                           torch.tensor(0, dtype=torch.long),
                                           torch.where(tensor == current_element,
                                                       torch.tensor(1, dtype=torch.long),
                                                       torch.tensor(2, dtype=torch.long)))
        sign_labels = output_matrix.flatten()

       


        return sign_labels




