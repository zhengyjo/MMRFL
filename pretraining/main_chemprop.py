import pandas as pd
from Utils.KMGCLConfig import KMGCLConfig
from Utils.BuildDatasetLoader import build_dataset_loader
from GraphModels.NG import GNNNodeEncoder, GNNGraphEncoder
from KMGCLModels.KMGCLModel import KMGCLModel
import torch
from SequenceModels.CNMRModel import CNMREncoderInterface
from SequenceModels.SmilesModel import SmilesEncoderInterface
from ImageModels.ImageModel import ImageEncoderInterface
from Utils.TrainEpoch import train_epoch

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data.utils import get_data_cl
from chemprop.data import get_data,get_task_names, MoleculeDataset, validate_dataset_type,MoleculeDataLoader
from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters
from chemprop.models import MoleculeModel,mpn

import os
import argparse
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='gin', help="Model type.")
parser.add_argument("--num_layer", type=int, default=5, help="Number of layers.")
parser.add_argument("--embed_dim", type=int, default=128, help="Embed dimension.")
parser.add_argument("--path", type=str, default='./pretraining/Utils/pretraining_data_demo.pkl', help="smiles file")
parser.add_argument("--graphMetric", type=str, default='smiles', help="graphMetric")
parser.add_argument("--nodeMetric", type=str, default='peak', help="nodeMetric")
parser.add_argument("--alpha", type=float, default=0, help="alpha")
# Parse and print the results
args = parser.parse_args()

def main():
    # Create the parser and add arguments
    out_name = "best_" + args.graphMetric + "_atom_alpha_" + str(args.alpha) + "_chemprop"
    print('Output name:%s' % out_name)
    if args.alpha == 1:
        out_name = "best_" + args.nodeMetric + "_atom_alpha_" + str(args.alpha) + "_chemprop"
        print('Output name:%s' % out_name)
    
    KMGCLConfig.graphMetric_method = args.graphMetric
    KMGCLConfig.nodeMetric_method = args.nodeMetric
    KMGCLConfig.alpha = args.alpha
    
    arguments = [
    '--data_path', args.path,
    '--dataset_type', 'kmgcl',
    '--smiles_columns','smiles',
    '--gpu','0'
    ]
    
    pass_args=TrainArgs().parse_args(arguments)

    # Generate train_dataset_loader and valid_dataset_loader
    train_dataset_loader = build_dataset_loader(KMGCLConfig,pass_args)

    # graph_model
#     nodeEncoder = GNNNodeEncoder(args.num_layer, args.embed_dim, JK="last", gnn_type=args.type, aggr='add').to(KMGCLConfig.device)
#     graph_model = GNNGraphEncoder(nodeEncoder, args.embed_dim, graph_pooling="add").to(KMGCLConfig.device)
    graph_model = mpn.MPN(pass_args).to(KMGCLConfig.device)

    # pre-trained models
    cnmr_model = CNMREncoderInterface()
    image_model = ImageEncoderInterface()
    smiles_model = SmilesEncoderInterface()

    device = KMGCLConfig.device
    model = KMGCLModel(graph_model=graph_model,
                       cnmr_model=cnmr_model.model,
                       image_model=image_model.model,
                       smiles_model=smiles_model.model,
                       config=KMGCLConfig).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=KMGCLConfig.lr, weight_decay=KMGCLConfig.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=KMGCLConfig.patience, factor=KMGCLConfig.factor
    )

    step = "epoch"

    best_loss = float('inf')
    for epoch in range(KMGCLConfig.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_dataset_loader, optimizer, lr_scheduler, step, KMGCLConfig.accuracies_req)

        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save(model.state_dict(), out_name + ".pt")
            torch.save(model.graph_encoder.model.state_dict(), out_name + "_encoder.pt")
            print("Saved Best Model!")

        print("\n")



if __name__ == "__main__":
    main()



