import torch

from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
            # Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))

    def forward(self, x):
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat([cls_tokens, x], axis=1)
        x += self.positions 

        return x 
    