import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers

class FFT_attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_conv = nn.Conv2d(dim*2, dim, 3, 1, 1)  # 处理实虚拼接通道
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, spatial_size=None):
        x = x.permute(0,2,3,1)
        B, H, W, C = x.shape
        
        x_fft = torch.fft.rfft2(x, dim=(1,2), norm='ortho')  # [B, H, W//2+1, C]
        
        real = x_fft.real.permute(0,3,1,2)  # [B, C, H, W//2+1]
        imag = x_fft.imag.permute(0,3,1,2)
        combined = torch.cat([real, imag], dim=1)  # [B, 2C, H, W//2+1]
        
        attn = self.sigmoid(self.attention_conv(combined))  # [B, C, H, W//2+1]
        attn = attn.permute(0,2,3,1)  # [B, H, W//2+1, C]
        
        filtered_fft = x_fft * attn
        
        x = torch.fft.irfft2(filtered_fft, s=(H,W), dim=(1,2), norm='ortho')
        x = x.permute(0,3,1,2)
        return x


class Vit_FFT_extractor(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(Vit_FFT_extractor, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.filter = FFT_attention(dim=dim)
    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm) + self.filter(x_norm)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 out_fratures,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias,padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_conv = nn.Conv2d(dim*2, dim, 3, 1, 1)  # 处理实虚拼接通道
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        B, H, W, C = x.shape
        
        x_fft = torch.fft.rfft2(x, dim=(1,2), norm='ortho')  # [B, H, W//2+1, C]
        
        real = x_fft.real.permute(0,3,1,2)  # [B, C, H, W//2+1]
        imag = x_fft.imag.permute(0,3,1,2)
        combined = torch.cat([real, imag], dim=1)  # [B, 2C, H, W//2+1]
        
        attn = self.sigmoid(self.attention_conv(combined))  # [B, C, H, W//2+1]
        attn = attn.permute(0,2,3,1)  # [B, H, W//2+1, C]
        
        filtered_fft = x_fft * attn

        x = torch.fft.irfft2(filtered_fft, s=(H,W), dim=(1,2), norm='ortho')
        x = x.permute(0,3,1,2)
        return x

class Vit_FFT(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(Vit_FFT, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor)
        self.filter = SpectralGatingNetwork(dim=dim)
    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm) + self.filter(x_norm)
        x = x + self.mlp(self.norm2(x))
        return x


class FFVit_block(nn.Module):
    def __init__(self,out_dim):
        super(FFVit_block, self).__init__()
        self.GlobalFeature = Vit_FFT(dim=out_dim, num_heads = 8)  
        self.FFN=nn.Conv2d(out_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect") 
    def forward(self, x):
        x1=self.GlobalFeature(x)
        return x1


import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import math

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 1, 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if scale == 1:
            m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        elif (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 1, 2^n and 3.')
        super(Upsample, self).__init__(*m)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)





class FFVit(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_out_ch=1,
                 num_layers=6,
                 embed_dim=24,
                 upscale=2):
        super(FFVit, self).__init__()
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = FFVit_block(embed_dim)
            self.layers.append(layer)

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride=1, padding=1, padding_mode='reflect')
        )

        self.downsample = nn.Sequential(
            nn.AvgPool2d(upscale, stride=upscale)  # 抗锯齿下采样
        )       

        self.embed=nn.Conv2d(embed_dim, embed_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")

        self.upsample = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)  # 双线性上采样
 
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_first = self.first_conv(x)
        x = self.downsample(x_first)
        x = self.embed(x)
        x_ps = x
        for layer in self.layers:
            x = layer(x)
        x = x_ps + x
        x = self.upsample(x) + x_first
        x = self.conv_last(x)
        return x
 