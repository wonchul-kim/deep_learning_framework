import torch
import torch.nn as nn
from functools import partial 
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block = nn.Identity()
        self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = x 
        if self._apply_shortcut:
            residual = self.shortcut(x)
        x = self.block(x)
        x += residual 
        
        return x 
    
    def _apply_shortcut(self):
        return self.in_channels != self.out_channels
    
# x = torch.ones((1, 1, 1, 1))
# print(x)
# resi = ResidualBlock(1, 1)
# print(x, resi(x))


class ConvNormActBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, norm=nn.BatchNorm2d, 
                act=nn.ReLU, **kwargs):
        
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size//2),
            norm(out_channels),
            act()
        )

conv3x3 = partial(ConvNormActBlock, kernel_size=3)
conv1x1 = partial(ConvNormActBlock, kernel_size=1)

# print(conv3x3)

class ResNetBasicBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion = expansion
        self.block = nn.Sequential(
                    conv3x3(in_channels, out_channels, bias=False, stride=downsampling),
                    conv3x3(in_channels, out_channels, bias=False, stride=downsampling,
                            act=nn.Identity),
        )
        self.shortcut = conv3x3(in_channels, self.expanded_channels, act=nn.Identity,
                                kernel_size=1, stride=downsampling, bias=False)
        
    @property
    def expanded_channels(self):
        return self.out_channels*self.expansion
    
    @property 
    def _apply_shortcut(self):
        return self.in_channels != self.expanded_channels

# print(ResNetBasicBlock(1, 1))
    

class BottleNeckBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion, downsampling, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels, kernel_size=1),
            conv3x3(in_channels, out_channels, kernel_size=3, stride=downsampling),
            conv3x3(in_channels, self.expanded_channels, kernel_size=1, act=nn.Identity),
        )
        
        
dummy = torch.ones((1, 32, 10, 10))

block = BottleNeckBlock(32, 64, 4, 1)
block(dummy).shape
print(block)
            
        