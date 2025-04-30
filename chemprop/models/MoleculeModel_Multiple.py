from typing import List, Union, Tuple
from logging import Logger
from argparse import Namespace

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import build_ffn, MultiReadout
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights

from chemprop.models import MoleculeModel
import collections

import ast

# from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
#     save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean


# New Modification
def overwrite_state_dict(
    loaded_param_name: str,
    model_param_name: str,
    loaded_state_dict: collections.OrderedDict,
    model_state_dict: collections.OrderedDict,
    logger: Logger = None,
) -> collections.OrderedDict:
    """
    Overwrites a given parameter in the current model with the loaded model.
    :param loaded_param_name: name of parameter in checkpoint model.
    :param model_param_name: name of parameter in current model.
    :param loaded_state_dict: state_dict for checkpoint model.
    :param model_state_dict: state_dict for current model.
    :param logger: A logger.
    :return: The updated state_dict for the current model.
    """
    debug = logger.debug if logger is not None else print

    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')

    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(
            f'Pretrained parameter "{loaded_param_name}" '
            f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
            f"model parameter of shape {model_state_dict[model_param_name].shape}."
        )

    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]

    return model_state_dict

def load_encoder_model(model: torch.nn,
    path: str,
    current_args: Namespace = None,
    cuda: bool = None,
    logger: Logger = None,
) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    loaded_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    #loaded_state_dict = loaded_encoder_model["state_dict"]
    model_state_dict = model.state_dict()
    
    encoder_param_names = [
            "encoder.encoder.0.W_i.weight",
            "encoder.encoder.0.W_h.weight",
            "encoder.encoder.0.W_o.weight",
            "encoder.encoder.0.W_o.bias"
        ]
    
    loaded_encoder_param_names = [
            "encoder.0.W_i.weight",
            "encoder.0.W_h.weight",
            "encoder.0.W_o.weight",
            "encoder.0.W_o.bias"
        ]
    
    for i in range(len(loaded_encoder_param_names)):
             model_state_dict = overwrite_state_dict(
                    loaded_encoder_param_names[i], encoder_param_names[i], loaded_state_dict, model_state_dict
                )
#     for param_name in encoder_param_names:
#                 model_state_dict = overwrite_state_dict(
#                     param_name, param_name, loaded_state_dict, model_state_dict
#                 )
            
    model.load_state_dict(model_state_dict)
    return model

class MoleculeModel_Multiple(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs,num_models,logger: Logger = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel_Multiple, self).__init__()
        
        self.logger = logger
        self.model_lst = nn.ModuleList([])
        self.coefficients = nn.ModuleList([])
        self.num_models = num_models
        #self.encoder_path = args.encoder_path.split(",")
        list_of_model = ast.literal_eval(args.encoder_path)
        self.encoder_path = list_of_model
        #self.encoder_path = args.encoder_path.split(",")
#         for model_idx in range(num_models):
#             temp =  MoleculeModel(args)
#             if args.encoder_path is not None:
#                 temp = load_encoder_model(model=temp,path=self.encoder_path[model_idx],current_args=args, logger=self.logger)
#             self.model_lst.append(temp.encoder.to(args.device))
        for model_idx in range(num_models):
            temp =  MoleculeModel(args)
            if args.encoder_path is not None:
                temp = load_encoder_model(model=temp,path=self.encoder_path[model_idx],current_args=args, logger=self.logger)
            self.model_lst.append(temp.to(args.device))
            readout = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size//3),  # First layer (input to hidden)
                nn.ReLU(),                           # Activation function
                nn.Linear(args.hidden_size//3, args.hidden_size//3//3),  # Second layer (hidden to output)
                nn.ReLU(), 
                nn.Linear(args.hidden_size//3//3, 1),
            )
            self.coefficients.append(readout)
        
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function

        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]

        self.is_atom_bond_targets = args.is_atom_bond_targets

        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets = args.atom_targets, args.bond_targets
            self.atom_constraints, self.bond_constraints = (
                args.atom_constraints,
                args.bond_constraints,
            )
            self.adding_bond_types = args.adding_bond_types

        self.relative_output_size = 1
        if self.multiclass:
            self.relative_output_size *= args.multiclass_num_classes
        if self.loss_function == "mve":
            self.relative_output_size *= 2  # return means and variances
        if self.loss_function == "dirichlet" and self.classification:
            self.relative_output_size *= (
                2  # return dirichlet parameters for positive and negative class
            )
        if self.loss_function == "evidential":
            self.relative_output_size *= (
                4  # return four evidential parameters: gamma, lambda, alpha, beta
            )

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = nn.Softplus()
            
        
        if self.is_atom_bond_targets:
            self.output_size = self.relative_output_size
        else:   
            self.output_size = self.relative_output_size * args.num_tasks,
            
    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """
        outputs_lst = []
        for idx in range(self.num_models):
            temp = self.model_lst[idx](
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
                constraints_batch,
                bond_types_batch,
            )
            embedding = self.model_lst[idx].encoder(batch)
            coefficient = self.coefficients[idx](embedding)
            outputs_lst.append(temp * coefficient)
            
        output = outputs_lst[0]
        
        for idx in (1,self.num_models-1):
            output = output + outputs_lst[idx]
            
        # Don't apply sigmoid during training when using BCEWithLogitsLoss
        if (
            self.classification
            and not (self.training and self.no_training_normalization)
            and self.loss_function != "dirichlet"
        ):
            if self.is_atom_bond_targets:
                output = [self.sigmoid(x) for x in output]
            else:
                output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.shape[0], -1, self.num_classes)
            )  # batch size x num targets x num classes per target
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(
                    output
                )  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Modify multi-input loss functions
        if self.loss_function == "mve":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, variances = torch.split(x, x.shape[1] // 2, dim=1)
                    variances = self.softplus(variances)
                    outputs.append(torch.cat([means, variances], axis=1))
                return outputs
            else:
                means, variances = torch.split(output, output.shape[1] // 2, dim=1)
                variances = self.softplus(variances)
                output = torch.cat([means, variances], axis=1)
        if self.loss_function == "evidential":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, lambdas, alphas, betas = torch.split(
                        x, x.shape[1] // 4, dim=1
                    )
                    lambdas = self.softplus(lambdas)  # + min_val
                    alphas = (
                        self.softplus(alphas) + 1
                    )  # + min_val # add 1 for numerical contraints of Gamma function
                    betas = self.softplus(betas)  # + min_val
                    outputs.append(torch.cat([means, lambdas, alphas, betas], dim=1))
                return outputs
            else:
                means, lambdas, alphas, betas = torch.split(
                    output, output.shape[1] // 4, dim=1
                )
                lambdas = self.softplus(lambdas)  # + min_val
                alphas = (
                    self.softplus(alphas) + 1
                )  # + min_val # add 1 for numerical contraints of Gamma function
                betas = self.softplus(betas)  # + min_val
                output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == "dirichlet":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    outputs.append(nn.functional.softplus(x) + 1)
                return outputs
            else:
                output = nn.functional.softplus(output) + 1

        return output
        