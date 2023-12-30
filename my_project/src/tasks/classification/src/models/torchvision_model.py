import torch.nn as nn

class TorchvisionModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = 