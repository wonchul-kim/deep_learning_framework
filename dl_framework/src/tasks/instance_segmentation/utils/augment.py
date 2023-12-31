import torch 
from torchvision.transforms import v2 as T

def get_transform(train, resize=None):
    transforms = []
    if resize:
        transforms.append(T.Resize(resize))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)