import torch 
import torch.nn as nn 
import einops
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, emb_size=768, use_conv=True):
        super().__init__()

        if use_conv:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e')
            )
        else:
            self.projection = nn.Sequential(
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                                                s1=patch_size, s2=patch_size),
                nn.Linear(patch_size*patch_size*in_channels, emb_size)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x) # batch num_patches 1d_patches
        
        cls_tokens = einops.repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1) # batch num_patches+1 1d_patches

        x += self.positions

        return x    


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()
        
        self.num_heads = num_heads
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        
        self.scale = (emb_size//num_heads)**(-0.5)
        self.attention_dropout = nn.Dropout(dropout)
        
        self.projection = nn.Linear(emb_size, emb_size)
        
        
    def forward(self, x, mask=None):
        
        # rearrange q, k, v into BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE/NUM_HEADS
        quries = einops.rearrange(self.queries(x), 
                            'b n (h d) -> b h n d', h=self.num_heads)
        keys = einops.rearrange(self.keys(x), 
                            'b n (h d) -> b h n d', h=self.num_heads)
        values = einops.rearrange(self.values(x), 
                            'b n (h d) -> b h n d', h=self.num_heads)
        
        # BATCH, HEADS, QUERY_LEN, KEY_LEN
        attention_score = torch.einsum('bhqd, bhkd -> bhqk', quries, keys)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            attention_score.mask_fill(~mask, fill_value)            
        
        attention_score = nn.functional.softmax(attention_score, dim=-1)*self.scale
        attention_score = self.attention_dropout(attention_score)
        
        out = torch.einsum('bhal, bhlv -> bhav', attention_score, values)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        
        return out
        

class ResidualAddBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 
        
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)
    

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, dropout=0, 
                    forward_expansion=4, forward_dropout=0, **kwargs):
        super().__init__(
            ResidualAddBlock(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(dropout)
                )
            ),
            ResidualAddBlock(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, dropout=forward_dropout),
                nn.Dropout(dropout)
                )
            ),
        )
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, num_classes=1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
        
class ViT(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768,
                    img_size=224, depth=12, num_classes=1000, **kwargs):
        super().__init__(
            PatchEmbedding(img_size, in_channels, patch_size, emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )
        
        
x = torch.randn((1, 3, 224, 224))

patch_size = 16
patches = einops.rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                                s1=patch_size, s2=patch_size)        

# patches_embedded = PatchEmbedding()(x)
# print(patches_embedded.shape)
# print(MultiHeadAttention()(patches_embedded).shape)

# patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape)

summary(ViT(), (3, 224, 224), device='cpu')
