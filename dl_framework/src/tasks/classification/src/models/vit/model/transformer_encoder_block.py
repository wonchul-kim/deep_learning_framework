import torch.nn as nn 
from .multi_head_attention import MultiHeadAttention

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 

    def forward(self, x, **kwargs):
        _x = x
        x = self.fn(x, **kwargs)

        return x + _x

class FeedForwardNN(nn.Sequential):
    def __init__(self, emb_size, expansion=4, dropout=0):
        super().__init__(
            nn.Linear(emb_size, expansion*emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion*emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, dropout=0, forward_expansion=4, forward_dropout=0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(dropout),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardNN(
                    emb_size, expansion=forward_expansion, dropout=forward_dropout),
                nn.Dropout(dropout),
                )
            )
        )
        
if __name__ == '__main__':
    import torch
    from patch_embedding import PatchEmbedding
    from transformer_encoder_block import TransformerEncoderBlock
    from torchsummary import summary

    x = torch.randn(8, 3, 224, 224)
    patch_embedding = PatchEmbedding()
    multi_head_attention = MultiHeadAttention()
    x = patch_embedding(x)
    x = multi_head_attention(x)
    transformer_encoder = TransformerEncoderBlock()
    summary(transformer_encoder, x.shape[1:], device='cpu')                         
