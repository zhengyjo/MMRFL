from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pickle

class GraphDataset(Dataset):
    def __init__(self, files, config, data_loading_function='gdata_loading_method1'):
        self.files = files
        self.graphs_path = config.graphs_path
        self.data_loading_function_name = data_loading_function

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.files[idx]

    def __getitem__(self, idx):
        graph_path = self.graphs_path + self.files[idx]
        graph_data = None

        if self.data_loading_function_name == 'gdata_loading_method1':
            graph_data = self.gdata_loading_method1(graph_path)
        return graph_data

    ### Methods to load molecular graphs###
    @staticmethod
    def gdata_loading_method1(graph_path):

        # load graph from pickle file
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)

        return data

    ###Collate function: collect and pad the batch data into same dims/shapes###
    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)


