import torch
import torch.nn as nn
import numpy as np
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from nnunetv2.utilities.decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
import torch.nn.functional as F

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embedding_dim, head_num):
#         super().__init__()

#         self.head_num = head_num
#         self.dk = (embedding_dim // head_num) ** (1 / 2)

#         self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
#         self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

#     def forward(self, x, mask=None):
#         qkv = self.qkv_layer(x)

#         query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
#         energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

#         if mask is not None:
#             energy = energy.masked_fill(mask, -np.inf)

#         attention = torch.softmax(energy, dim=-1)

#         x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

#         x = rearrange(x, "b h t d -> b t (h d)")
#         x = self.out_attention(x)

#         return x

# class MLP(nn.Module):
#     def __init__(self, embedding_dim, mlp_dim):
#         super().__init__()

#         self.mlp_layers = nn.Sequential(
#             nn.Linear(embedding_dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(mlp_dim, embedding_dim),
#             nn.Dropout(0.1)
#         )

#     def forward(self, x):
#         x = self.mlp_layers(x)

#         return x

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, embedding_dim, head_num, mlp_dim):
#         super().__init__()

#         self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
#         self.mlp = MLP(embedding_dim, mlp_dim)

#         self.layer_norm1 = nn.LayerNorm(embedding_dim)
#         self.layer_norm2 = nn.LayerNorm(embedding_dim)

#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         _x = self.multi_head_attention(x)
#         _x = self.dropout(_x)
#         x = x + _x
#         x = self.layer_norm1(x)

#         _x = self.mlp(x)
#         x = x + _x
#         x = self.layer_norm2(x)

#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
#         super().__init__()

#         self.layer_blocks = nn.ModuleList(
#             [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

#     def forward(self, x):
#         for layer_block in self.layer_blocks:
#             x = layer_block(x)

#         return x

# class ViT(nn.Module):
#     def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
#                  block_num, patch_dim, classification=True, num_classes=1):
#         super().__init__()

#         self.patch_dim = patch_dim
#         self.classification = classification
#         self.num_tokens = (img_dim // patch_dim) ** 2
#         self.token_dim = in_channels * (patch_dim ** 2)

#         self.projection = nn.Linear(self.token_dim, embedding_dim)
#         self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

#         self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

#         self.dropout = nn.Dropout(0.1)

#         self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

#         if self.classification:
#             self.mlp_head = nn.Linear(embedding_dim, num_classes)

#     def forward(self, x):
#         img_patches = rearrange(x,
#                                 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
#                                 patch_x=self.patch_dim, patch_y=self.patch_dim)
#         batch_size, tokens, _ = img_patches.shape

#         project = self.projection(img_patches)
#         token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
#                        batch_size=batch_size)

#         patches = torch.cat([token, project], dim=1)
#         patches += self.embedding[:tokens + 1, :]
#         x = self.dropout(patches)
#         x = self.transformer(x)
#         x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
#         return x

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
        # print("load_from:",weights)
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
# model = ViT()

# pretrained_weights = np.load('/home/hcy/nnUNet/DATASET/nnUNet_raw/ViT-B_16.npz')
# model.load_from(pretrained_weights)

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT()
        self.vit.load_from(np.load('/opt/app/model/nnUNet/DATASET/nnUNet_raw/ViT-B_16.npz'))

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = x.view(x.shape[0],1,768,768)
        x = torch.repeat_interleave(x, 3, dim=1)
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x = self.vit(x)
        x = x.view(x.size(0),1024,24,18)
        x = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)
        self.conv2 = nn.Conv2d(int(out_channels * 1 / 2), class_num, kernel_size=1)
        self.conv3 = nn.Conv2d(int(out_channels), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        skips = []
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        skips.append(self.conv3(x))
        x = self.decoder3(x, x1)
        skips.append(self.conv2(x))
        x = self.decoder4(x)
        x = self.conv1(x)
        skips.append(x)

        return x,skips


class TransUNetsys(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x,skips = self.decoder(x, x1, x2, x3)

        return x,skips

class TransUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 input_image_dim = 384
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.input_img_dim = input_image_dim
        self.strides = strides[:-1]
        self.use_deep_supervision = deep_supervision
        # self.encoder = TransUNetEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
        #                                 n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
        #                                 dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
        #                                 nonlin_first=nonlin_first,input_image_dim=input_image_dim)
        self.model = TransUNetsys(input_image_dim, input_channels, 128, 4, 512, 8, 16, num_classes)
        # self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
        #                            nonlin_first=nonlin_first)
        self.decoder = TransUNetsys(input_image_dim, input_channels, 128, 4, 512, 8, 16, num_classes)

    def forward(self, x):
        original_size = x.shape[-2:]
        input_tensor = F.interpolate(x, size=(self.input_img_dim, self.input_img_dim), mode='bilinear', align_corners=False)
        res_,skips = self.model(input_tensor)

        if self.use_deep_supervision:
            skips = skips[::-1]
            # skips = self.encoder(input_tensor)
            # output_tensors = self.decoder(skips)
            res = []
            scale = 1
            for tensor,stride in zip(skips,self.strides):
                if type(stride) == int:
                    scale *= stride
                elif type(stride) in (tuple, list):
                    scale *= stride[0]
                tensor = F.interpolate(tensor,(original_size[0]//scale,original_size[1]//scale), mode='bilinear', align_corners=False)
                res.append(tensor)
            return res
        else:
            tensor = F.interpolate(res_, size=original_size, mode='bilinear', align_corners=False)
            return tensor

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)




if __name__ == '__main__':
    ### test Encoder
    # encoder = Encoder(img_dim=512, in_channels=1, out_channels=64, head_num=8, mlp_dim=2048, block_num=12, patch_dim=16)
    # print(sum(p.numel() for p in encoder.parameters()))
    # print(encoder(torch.rand(1, 1, 512, 512))[0].shape)
    ### test TransUNetEncoder
    # encoder = TransUNetEncoder(input_channels=1,
    #                            n_stages=4,
    #                            features_per_stage=[32, 64, 128, 256],
    #                            conv_op=nn.Conv2d,
    #                            kernel_sizes=[3, 3, 3, 3],
    #                            strides=[1, 2, 2, 2],
    #                            n_conv_per_stage=[2, 2, 2, 2],
    #                            conv_bias=False,
    #                            norm_op=nn.InstanceNorm2d,
    #                            norm_op_kwargs={'eps': 1e-5, 'affine': True},
    #                            dropout_op=nn.Dropout2d,
    #                            dropout_op_kwargs={'p': 0.1},
    #                            nonlin=nn.LeakyReLU,
    #                            nonlin_kwargs={'inplace': True},
    #                            return_skips=True,
    #                            nonlin_first=False,
    #                            pool='conv',input_image_dim=256)
    # print(sum(p.numel() for p in encoder.parameters()))
    # print(encoder(torch.rand(1, 1, 256, 256)).shape)
    ### test TransUNet
    model = TransUNet(input_channels=1,
                      n_stages=4,
                      features_per_stage=[32, 64, 128, 256],
                      conv_op=nn.Conv2d,
                      kernel_sizes=[3, 3, 3, 3],
                      strides=[1, 2, 2, 2],
                      n_conv_per_stage=2,
                      num_classes=6,
                      n_conv_per_stage_decoder=2,
                      conv_bias=False,
                      norm_op=nn.InstanceNorm2d,
                      norm_op_kwargs={'eps': 1e-5, 'affine': True},
                      dropout_op=nn.Dropout2d,
                      dropout_op_kwargs={'p': 0.1},
                      nonlin=nn.LeakyReLU,
                      nonlin_kwargs={'inplace': True},
                      deep_supervision=True,
                      nonlin_first=False,
                      input_image_dim=384)
    print(sum(p.numel() for p in model.parameters()))
    for res in  model(torch.rand(1, 1, 512, 448)):
        print(res.shape)



# class TransUNetEncoder(nn.Module):
#     def __init__(self,
#                  input_channels: int,
#                  n_stages: int,
#                  features_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  conv_op: Type[_ConvNd],
#                  kernel_sizes: Union[int, List[int], Tuple[int, ...]],
#                  strides: Union[int, List[int], Tuple[int, ...]],
#                  n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  conv_bias: bool = False,
#                  norm_op: Union[None, Type[nn.Module]] = None,
#                  norm_op_kwargs: dict = None,
#                  dropout_op: Union[None, Type[_DropoutNd]] = None,
#                  dropout_op_kwargs: dict = None,
#                  nonlin: Union[None, Type[torch.nn.Module]] = None,
#                  nonlin_kwargs: dict = None,
#                  return_skips: bool = False,
#                  nonlin_first: bool = False,
#                  pool: str = 'conv',
#                  input_image_dim =256
#                  ):
#         super().__init__()
#         if isinstance(kernel_sizes, int):
#             kernel_sizes = [kernel_sizes] * n_stages
#         if isinstance(features_per_stage, int):
#             features_per_stage = [features_per_stage] * n_stages
#         if isinstance(n_conv_per_stage, int):
#             n_conv_per_stage = [n_conv_per_stage] * n_stages
#         if isinstance(strides, int):
#             strides = [strides] * n_stages
#         assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
#         assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
#         assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
#         assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
#                                              "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

#         self.img_dim = input_image_dim
#         self.patch_size = 1
#         stages = []
#         for s in range(n_stages):
#             stage_modules = []
#             if pool == 'max' or pool == 'avg':
#                 if (isinstance(strides[s], int) and strides[s] != 1) or \
#                         isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
#                     stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
#                 conv_stride = 1
#             elif pool == 'conv':
#                 conv_stride = strides[s]
#             else:
#                 raise RuntimeError()

#             stage_modules.append(StackedConvBlocks(
#                 n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
#                 conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
#             ))
#             if type(strides[s]) == int:
#                 self.img_dim = self.img_dim // strides[s]
#             elif type(strides[s]) in (tuple, list):
#                 self.img_dim = self.img_dim // strides[s][0]
#             stages.append(nn.Sequential(*stage_modules))
#             input_channels = features_per_stage[s]
        
#         self.vit = ViT(self.img_dim, features_per_stage[-1], features_per_stage[-1], 4, 512, 8, self.patch_size, classification=False)
#         self.conv2 = nn.Conv2d(features_per_stage[-1],features_per_stage[-1], kernel_size=3, stride=1, padding=1)
#         self.norm2 = nn.BatchNorm2d(features_per_stage[-1])
#         self.stages = nn.Sequential(*stages)
#         self.relu = nn.ReLU(inplace=True)
#         self.output_channels = features_per_stage
#         self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
#         self.return_skips = return_skips

#         # we store some things that a potential decoder needs
#         self.conv_op = conv_op
#         self.norm_op = norm_op
#         self.norm_op_kwargs = norm_op_kwargs
#         self.nonlin = nonlin
#         self.nonlin_kwargs = nonlin_kwargs
#         self.dropout_op = dropout_op
#         self.dropout_op_kwargs = dropout_op_kwargs
#         self.conv_bias = conv_bias
#         self.kernel_sizes = kernel_sizes
#     def forward(self, x):
#         skips = []
#         for i in range(len(self.stages)):
#             x = self.stages[i](x)
#             if self.return_skips:
#                 if i == len(self.stages) - 1:
#                     x = self.vit(x)
#                     x = rearrange(x, "b (x y) c -> b c x y", x=self.img_dim, y=self.img_dim)
#                     x = self.relu(self.norm2(self.conv2(x)))
#                 skips.append(x)
#         return skips