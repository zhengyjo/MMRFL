import torch
from torch import nn

class MlpBlock(nn.Module):
    def __init__(self, channels):
        super(MlpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, input):
        return input + self.block(input)


class CnnBlock_2(nn.Module):
    def __init__(self, channels):
        super(CnnBlock_2, self).__init__()
        self.layer = nn.Conv1d(
            in_channels=1,
            out_channels=channels,
            kernel_size=400,
            stride=200,
            padding=200
        )

    def forward(self, input):
        output_1 = self.layer(input.unsqueeze(1))
        return output_1


class CNMRModel(nn.Module):

    ### Pretrained CNMRModel input channels is 4000.
    def __init__(self, input_channels=4000, nmr_output_channels=768, channels=32):
        super(CNMRModel, self).__init__()

        hidden_channels = channels * 21

        self.feature_dim = nmr_output_channels

        self.model = nn.Sequential(
            CnnBlock_2(channels),
            nn.ReLU(),
            nn.Flatten(),
            MlpBlock(hidden_channels),
            nn.ReLU(),
            MlpBlock(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, nmr_output_channels)
        )

    def forward(self, input):
        features = self.model(input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


class CNMREncoderInterface(nn.Module):
    def __init__(self,
                 model_load_pretrained='./pretraining/SequenceModels/PretrainedWeights/nmr_model_weights.pth'):
        super(CNMREncoderInterface, self).__init__()
        self.model = CNMRModel()
        state_dict = torch.load(model_load_pretrained)
        self.model.load_state_dict(state_dict)

