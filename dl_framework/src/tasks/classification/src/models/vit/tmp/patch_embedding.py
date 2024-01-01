import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

### input
batch_size = 8
img_size = 224 # input image size
x = torch.randn(batch_size, 3, img_size, img_size)
print('* x: ', x.shape)

patch_size = 16
num_patches = int(img_size*img_size/(patch_size*patch_size))
in_channels = 3
emb_size = 768

### 1. bchw -> b*num_patches*(p*p*c)
### case 1) 
patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size, p2=patch_size)
print('* patches: ', patches.shape)

### case 2)

projection = nn.Sequential(
    nn.Conv2d(in_channels, emb_size,
              kernel_size=patch_size, stride=patch_size),
    Rearrange('b e h w -> b (h w) e')
)
summary(projection, x.shape[1:], device='cpu')

### 2. 
### projection
proj_x = projection(x)
print('* projected x: ', proj_x.shape)

### 
cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
print("* cls_token: ", cls_token.shape)
print("* positions: ", positions.shape)

### repeat cls_token as batch_size
cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)
print('* repeated cls_token: ', cls_tokens.shape)

### concatenate cls_tokens and proj_x
cat_x = torch.cat([cls_tokens, proj_x], axis=1)

### add positions
cat_x += positions

print('* cat_x: ', cat_x.shape)

