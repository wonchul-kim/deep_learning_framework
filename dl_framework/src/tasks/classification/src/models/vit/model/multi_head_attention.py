import torch

from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scale = self.emb_size**(1/2)
        attn = F.softmax(energy/scale, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhal, bhlv -> bhav', attn, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out 
    
    if __name__ == '__main__':
        from patch_embedding import PatchEmbedding
        from multi_head_attention import MultiHeadAttention
        from torchsummary import summary

        x = torch.randn(8, 3, 224, 224)
        patch_embedding = PatchEmbedding()
        x = patch_embedding(x)
        print('patch_embedding: ', x.shape)
        
        multi_head_attention = MultiHeadAttention()
        summary(multi_head_attention, x.shape[1:], device='cpu')
