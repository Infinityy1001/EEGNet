import torch
from EEGNet import EEGNet

model = torch.load('TEST_PYTORCH/model/model.pth',  weights_only=False)
print(model.eval())