# MULTIMODAL FUSION WITH RELATIONAL LEARNING FOR MOLECULAR PROPERTY PREDICTION

## Features

- Use relational metric to capture the complex relationships among molecules
- Explore how fusion helps improve the representation

## When doing the pretraining, please prepare the following folders:
####  `pickle` Files for SMILES
These files contain serialized molecular data in the form of **SMILES** (Simplified Molecular Input Line Entry System) strings, providing a compact textual representation of molecular structures.

####  `mapping.csv`
A central CSV file that maps each molecule to the corresponding data files. It tells the model where to find:
- The graph representation (from RDKit)
- C-NMR data (From nmrshiftdb2)
- Molecular fingerprint (from RDKit)
- Molecular image (from RDKit)

####  `graph_hyb/` Folder
Contains molecular **graph representations** (e.g., adjacency matrices or edge lists) for use in graph-based learning models.

####  `cnmr/` Folder
Holds **C-NMR (Carbon Nuclear Magnetic Resonance)** data and peak information for each molecule.

####  `fingerprint/` Folder
Stores **molecular fingerprints**—bit vector representations capturing structural features of the molecules.

####  `image/` Folder
Contains **visual images** of molecular structures, which may be used in image-based modeling or for visualization purposes.

Demo folders and files are provided in pretraining/pretraining_data 

## Usage

Before running pretraining, please download the model.ckpt file for the Img2mol[1] model (~2.4GB) from the link: [Img2mol](https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view)],
and then put it under /pretraining/ImageModels/PretrainedWeights/

[1] D.-A. Clevert, T. Le, R. Winter, F. Montanari, *Chem. Sci.*, 2021, DOI: [10.1039/D1SC01839F](https://doi.org/10.1039/D1SC01839F)

Run the pretraining:

```sh
python pretraining/main_chemprop.py --data_path [directory of pretraining dataset] --graph_metric ['smiles' or 'image' or 'fingerprint'or 'nmr' or 'fusion_average' ] --nodeMetric ['peak'] --alpha [0 or 1]
```
Alpha is a parameter to adjust the weight of different modalities in pretraining (Early Fusion).
If you use 'smiles'or 'image' or 'fingerprint'or 'nmr' by itself, please set alpha to 0.
If you use 'peak' by itself, please set alpha to 1.

Finetuning with unimodality or early fusion:

```sh
python finetune_updated.py --dir [directory of dataset] --dataset_type [classification or regression] --seed [Seed Number] --gpu [gpu number] --encoder_path [pretrained weight for encoder]
```
Example:
python finetune_updated.py --dir data/bace.csv --dataset_type classification --seed 19 --gpu 0 --encoders pretrained_weight/smiles_weight.pt 

Finetuning with Intemediate Fusion:

```sh
python finetune_multi_intermediate.py --dir [directory of dataset] --dataset_type [classification or regression] --seed [Seed Number] --gpu [gpu number] --encoder_path [List of pretrained weights for Graph encoder in each modality]
```

Finetuning with Late Fusion:

```sh
python finetune_multi.py --dir [directory of dataset] --dataset_type [classification or regression] --seed [Seed Number] --gpu [gpu number] --encoder_path [List of pretrained weight for Graph encoder in each modality]
```

Example:
python finetune_multi_intermediate.py --dir data/bace.csv --dataset_type classification --seed 19 --gpu 0 --encoders "['./pretrained_weight/smiles.pt','./pretrained_weight/image.pt','./pretrained_weight/nmr.pt','./pretrained_weight/fingerprint.pt','./pretrained_weight/peak.pt']" 


# MMRFL
