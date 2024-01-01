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

from model.patch_embedding import PatchEmbedding

### input
batch_size = 8
img_size = 224 # input image size
x = torch.randn(batch_size, 3, img_size, img_size)
print('* x: ', x.shape)

patch_size = 16
num_patches = int(img_size*img_size/(patch_size*patch_size))
in_channels = 3
emb_size = 768

### patch embedding
patch_embedding = PatchEmbedding()
x = patch_embedding(x)
print('* patch_embedding: ', x.shape)

### Multi Head Attention 
num_heads = 8

###
keys = nn.Linear(emb_size, emb_size)
queries = nn.Linear(emb_size, emb_size)
values = nn.Linear(emb_size, emb_size)

k, q, v = keys(x), queries(x), values(x)
print("* keys, queries, values: ", k.shape, q.shape, k.shape)

keys = rearrange(k, 'b n (h d) -> b h n d', h=num_heads)
queries = rearrange(q, 'b n (h d) -> b h n d', h=num_heads)
values = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
print("* Heads keys, queries, values: ", keys.shape, queries.shape, keys.shape)

### queires*keys := Q.matmul(K.T)
energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
# energy = torch.matmul(queries, keys.transpose(2, 3))
print('* energy: ', energy.shape)

### attention score 
scale = emb_size**(1/2)
attention_score = F.softmax(energy, dim=-1)/scale 
print("* attetion score: ", attention_score.shape)

### attention score * value
out = torch.einsum('bhal, bhlv -> bhav', attention_score, values)
print("* out: ", out.shape)

out = rearrange(out, 'b h n d -> b n (h d)')
print("* out: ", out.shape)





