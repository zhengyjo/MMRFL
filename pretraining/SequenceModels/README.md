# 1. Create CNMRModel and Load pretrained weights

### Load the state dictionary from the checkpoint file
state_dict = torch.load('SequenceModels/PretrainedWeights/nmr_model_weights.pth')

### Remove the "logit_scale" key from the state dictionary
del state_dict['logit_scale']

### Create an instance of your model
nmr_model = CNMRModel()

### Load the modified state dictionary into the model
nmr_model.load_state_dict(state_dict, strict=False)


