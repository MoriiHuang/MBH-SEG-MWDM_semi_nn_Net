import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from nnunetv2.utilities.decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
import torch.nn.functional as F


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def show_weight(weight_path):
    vitfile =np.load(weight_path, allow_pickle=True)
    vitfile = vitfile.files
    print(vitfile)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attention(x, x, x)[0]

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def load_from(self, weights, n_block):
        ROOT = f'Transformer/encoderblock_{n_block}'
        with torch.no_grad():
            self.norm1.weight.copy_(np2th(weights[f'{ROOT}/LayerNorm_0/scale']))
            self.norm1.bias.copy_(np2th(weights[f'{ROOT}/LayerNorm_0/bias']))
            self.norm2.weight.copy_(np2th(weights[f'{ROOT}/LayerNorm_2/scale']))
            self.norm2.bias.copy_(np2th(weights[f'{ROOT}/LayerNorm_2/bias']))
            in_proj_weight = torch.stack((np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/query/kernel']), np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/key/kernel']), np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/value/kernel'])),dim=0)
            in_proj_bias = torch.stack((np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/query/bias']), np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/key/bias']), np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/value/bias'])),dim=0)
            self.attn.attention.in_proj_weight.copy_(in_proj_weight.view(self.attn.attention.in_proj_weight.shape))
            self.attn.attention.in_proj_bias.copy_(in_proj_bias.view(self.attn.attention.in_proj_bias.shape))
            self.attn.attention.out_proj.weight.copy_(np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/out/kernel']).view(self.attn.attention.out_proj.weight.shape))
            self.attn.attention.out_proj.bias.copy_(np2th(weights[f'{ROOT}/MultiHeadDotProductAttention_1/out/bias']).view(self.attn.attention.out_proj.bias.shape))
            self.mlp.fc1.weight.copy_(np2th(weights[f'{ROOT}/MlpBlock_3/Dense_0/kernel']).t())
            self.mlp.fc1.bias.copy_(np2th(weights[f'{ROOT}/MlpBlock_3/Dense_0/bias']))
            self.mlp.fc2.weight.copy_(np2th(weights[f'{ROOT}/MlpBlock_3/Dense_1/kernel']).t())
            self.mlp.fc2.bias.copy_(np2th(weights[f'{ROOT}/MlpBlock_3/Dense_1/bias']))

class ViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # cls_token_final = x[:, 0]
        # logits = self.head(cls_token_final)
        return x[:,1:]
    def load_from(self, weights):
        with torch.no_grad():
            self.patch_embed.proj.weight.copy_(np2th(weights['embedding/kernel'], conv=True).reshape(self.patch_embed.proj.weight.shape))
            self.patch_embed.proj.bias.copy_(np2th(weights['embedding/bias']))
            self.pos_embed.copy_(np2th(weights['Transformer/posembed_input/pos_embedding']))
            self.norm.weight.copy_(np2th(weights['Transformer/encoder_norm/scale']))
            self.norm.bias.copy_(np2th(weights['Transformer/encoder_norm/bias']))
            for i, block in enumerate(self.blocks):
                block.load_from(weights, i)
            self.head.weight.copy_(np2th(weights['head/kernel']).t())
            self.head.bias.copy_(np2th(weights['head/bias']))
# 创建 ViT 模型实例
model = ViT()

pretrained_weights = np.load('/home/hcy/nnUNet/DATASET/nnUNet_raw/ViT-B_16.npz')
model.load_from(pretrained_weights)

# 将模型设置为评估模式
model.eval()

# 创建一个随机输入
input_tensor = torch.randn(3, 3, 384, 384)

# 获取模型预测结果
with torch.no_grad():
    output = model(input_tensor)

print(output.shape)



# show_weight('/home/hcy/nnUNet/DATASET/nnUNet_raw/ViT-B_16.npz') 