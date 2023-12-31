import torch
import torch.nn as nn
import torchvision 

class TorchvisionModel(nn.Module):
    def __init__(self, model_name, in_channels, num_classes, pretrained=True):
        super().__init__()
        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(in_channels, self.model.conv1.out_channels,kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def load_weights(self, weights):
        self.model.load_state_dict(torch.load(weights)['model'])

if __name__ == '__main__':
    model = TorchvisionModel('resnet18', 1, 10)


