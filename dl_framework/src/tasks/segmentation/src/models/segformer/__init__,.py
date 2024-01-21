import torch 
import torch.nn as nn 
from einops import rearrange

class LayerNorm2d(nn.LayerNorm):
    """
    Since `torch.nn.LayerNorm` gets the input of channel-last, `b,h,w,c`,
    Need to change the order of channel.
    """
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> h c h w')
        
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=patch_size, 
                      stride=overlap_size,
                      padding=patch_size//2,
                      bias=False),
            LayerNorm2d(out_channels),
        )    

# We all know attention has a square complexity O(N^2) 
# where N=H*W in our case. We can reduce N by a factor of R, 
# the complexity becomes O(N^2/R). 
# One easy way is to flat the spatial dimension and use a linear layer.
r = 4
channels = 8
x = torch.randn((1, channels, 64, 64))
_, _, h, w = x.shape

# flat the spatial dimension
x = rearrange(x, 'b c h w -> b (h w) c') # [1, 4096, 8]
# reduce by r and rearrange it into channel 
x = rearrange(x, 'b (hw r) c -> b hw (c r)', r=r) # [1, 1024, 32]

x = nn.Linear(channels*r, channels)(x) # [1, 1024, 4]
half_r = r//2
x = rearrange(x, "b (h w) c -> b c h w", h=h//half_r) # [1, 8, 32, 32]

print(x.shape) 

# We have reduced the spatial size by r=4, so by 2 on each dimension 
# (height and width). If you think about it, you can use a convolution 
# layer with a kernel_size=r and a stride=r to achieve the same effect.