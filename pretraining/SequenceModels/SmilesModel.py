import torch
from torch import nn
import numpy as np
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaConfig

# Modified from https://github.com/Qihoo360/CReSS/blob/master/model/model_smiles.py

class SmilesModel(nn.Module):
    def __init__(self,
                 roberta_tokenizer_path=None,
                 smiles_maxlen=300,
                 vocab_size=181,
                 max_position_embeddings=505,
                 num_attention_heads=12,
                 num_hidden_layers=6,
                 type_vocab_size=1,
                 feature_dim=768,
                 **kwargs
                 ):
        super(SmilesModel, self).__init__(**kwargs)
        self.smiles_maxlen = smiles_maxlen
        self.feature_dim = feature_dim
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            hidden_size=self.feature_dim
        )

        self.model = RobertaModel(config=self.config)
        self.dense = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, input):
        hidden_states = self.model(input[:,:,0], input[:,:,1])[0][:, 0]
        features = self.dense(hidden_states)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

class SmilesEncoderInterface(nn.Module):
    def __init__(self,
                 model_tokenizer='./pretraining/SequenceModels/PretrainedWeights/tokenizer-smiles-roberta-1e',
                 model_load_pretrained='./pretraining/SequenceModels/PretrainedWeights/smiles_model_weights.pth',
                 ):
        super(SmilesEncoderInterface, self).__init__()
        self.model = SmilesModel(roberta_tokenizer_path=model_tokenizer)
        state_dict = torch.load(model_load_pretrained)
        del state_dict['logit_scale']
        self.model.load_state_dict(state_dict, strict=False)

