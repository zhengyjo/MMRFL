import torch

class KMGCLConfig:
    debug = True
    project_path = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # Parameters for Build Dataset Loader
    batch_size = 128
    random_seed = 42
    shuffle = True
    drop_last = True # It has to be true
    graphMetric_method = 'smiles'
    dataset_file = '/home/zhengyjo/M3-KMGCL-ZZ/Utils/dataset_nmrshiftdb2_modified_1204.csv'
    graphs_path = "/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/graph_hyb/"
    cnmr_path = "/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/cnmr/"
    fingerprint_path = '/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/fingerprint/'
    image_path = '/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/image/'
    smiles_path= '/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/smiles/'

    # Parameters for training
    lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 200

    # alpha * mr_loss + (1-alpha) * element_loss
    alpha = 0.5
    smiles_model_tokenizer='/home/zhengyjo/M3-KMGCL-ZZ/SequenceModels/PretrainedWeights/tokenizer-smiles-roberta-1e'
    cnmr_temperature = 1e-5
    cnmr_diff_temperature = 1e1

    # Parameters for Molecular Representation 2 Model
    mr2_name = "graph"
    mr2_model_name = "gin"
    mr2_model_embedding = 128
    mr2_model_pretrained = False
    mr2_model_trainable = True

    # Parameters for Projection
    num_projection_layers = 1
    projection_dim = [300]
    dropout = 0.1

    # Paramerts for CMRP Model
    temperature = 1.0

    #top accuracies
    accuracies_req = [1]
