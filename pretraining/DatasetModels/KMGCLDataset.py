from torch.utils.data import Dataset
from chemprop.features import get_features_generator
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.features import is_explicit_h, is_reaction, is_adding_hs, is_mol, is_keeping_atom_map
from chemprop.data.data import construct_molecule_batch

class KMGCLDataset(Dataset):
    def __init__(self, smile_input_dataset,graph_dataset, peak_dataset, nmr_dataset, image_dataset, smiles_dataset, fingerprint_dataset, config):
        self.smiles_input_dataset,self.graph_dataset, self.peak_dataset, self.nmr_dataset, self.image_dataset, self.smiles_dataset, self.fingerprint_dataset = \
            smile_input_dataset,graph_dataset, peak_dataset, nmr_dataset, image_dataset, smiles_dataset, fingerprint_dataset

        self.data_types = ['smiles_input','graph', 'peak', 'nmr', 'image', 'smiles', 'fingerprint']

    def __len__(self):
        return len(self.graph_dataset)

    # Zhengyang's modification based on molecule Dataset
    def get_data_and_filename(self, data_type, idx):
        data = getattr(self, f'{data_type}_dataset')[idx]
        if data_type != 'smiles_input':
            file_name = getattr(self, f'{data_type}_dataset').get_sample_name(idx)
            return {data_type: data, f'{data_type}_filename': file_name}
        else:
            return {data_type: data}

    def __getitem__(self, idx):
        return {key: value for data_type in self.data_types for key, value in self.get_data_and_filename(data_type, idx).items()}

    def collate_fn(self, batch):
        collated_batch = {}
        for data_type in self.data_types:
            data_batch = [item[data_type] for item in batch]
            if data_type != 'smiles_input':
                collated_batch[data_type] = getattr(self, f'{data_type}_dataset').collate_fn(data_batch)
            #collated_batch[f'{data_type}_names'] = [item[f'{data_type}_filename'] for item in batch]
            else:
                collated_batch[data_type] = construct_molecule_batch(data_batch)

        return collated_batch


