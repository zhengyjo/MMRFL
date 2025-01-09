# MULTIMODAL FUSION WITH RELATIONAL LEARNING FOR MOLECULAR PROPERTY PREDICTION

## Features

- Use relational metric to capture the complex relationships among molecules
- Explore how fusion helps improve the representation


## Usage

Run the pretraining:

```sh
python pretraining/main_chemprop.py --data_path [directory of pretraining dataset]
```

Finetuning with unimodality or early fusion:

```sh
python finetune_updated.py --data_path [directory of dataset] --encoder_path [pretrained weight for encoder]
```

Finetuning with Intemediate Fusion:

```sh
python finetune_multi_intermediate.py --data_path [directory of dataset] --encoder_path [List of pretrained weights for Graph encoder in each modality]
```

Finetuning with Late Fusion:

```sh
python finetune_multi.py --data_path [directory of dataset] --encoder_path [List of pretrained weight for Graph encoder in each modality]
```


# MMRFL
