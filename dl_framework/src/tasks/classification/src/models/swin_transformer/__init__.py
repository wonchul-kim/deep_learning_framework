import torch 
import torch.nn as nn 
from torchsummary import summary
import einops 
from einops.layers.torch import Rearrange, Reduce

# class PatchPartition(nn.Module):
#     def __init__(self, patch_size: int = 4):
#         super().__init__()
#         self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
#         self.norm = nn.LayerNorm(96)

#     def forward(self, x):
#         x = self.proj(x)                  # [B, 96, 56, 56]
#         x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
        
class PatchPartition(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, emb_size=96):
        super().__init__()
        
        self.projection = nn.Sequential(
                            nn.Conv2d(in_channels, emb_size, 
                                      kernel_size=patch_size,
                                      stride=patch_size),
                            Rearrange('b e (h) (w) -> b (h w) e')
        )
        
    def forward(self, x):
        x = self.projection(x)
        
        return x

    
PP = PatchPartition()
summary(PP, (3, 224, 224), device='cpu')