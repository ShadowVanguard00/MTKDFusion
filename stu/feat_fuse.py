import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, 
                     padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class CSAM(nn.Module):
    def __init__(
            self,
            in_chans: int,
            reduction: int = 16,
            kernel_size: int = 7,
            min_channels: int = 8,
    ):
        super(CSAM, self).__init__()
        hidden_chans = max(in_chans // reduction, min_channels)
        self.mlp_chans = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, in_chans, kernel_size=1, bias=False),
        )
        self.space_attn_1 = nn.Conv2d(in_chans, 1, kernel_size=1, padding=0, bias=False) 
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_x_s = x.mean((2, 3), keepdim=True)
        max_x_s = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        avg_x_s_new = (self.mlp_chans(avg_x_s)).squeeze(-1)
        max_x_s_new = (self.mlp_chans(max_x_s)).squeeze(-1)
        attn = self.gate(torch.matmul(avg_x_s_new, max_x_s_new.transpose(1, 2)))
        x_flatten = x.view(B, C, -1) 
        x_flatten = torch.matmul(attn, x_flatten)
        x = x_flatten.view(B, C, H, W)
        x_space_1 = self.space_attn_1(x)
        spital_attn = self.gate(x_space_1)
        return spital_attn

class CAFM(nn.Module):
    def __init__(self, in_c, dim, num_heads=4, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim   
        self.num_heads = num_heads  
        self.kernel_size = kernel_size  
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads 
        self.scale = self.head_dim ** -0.5 
         
        self.v = nn.Linear(dim, dim)
       
        self.attn_fg = CSAM(in_chans=in_c)
        self.attn_bg = CSAM(in_chans=in_c)
    
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
 
        self.input_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1)  
        )
        self.fg_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1)  
        )
        self.bg_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1)  
        )
        self.relu = nn.ReLU(inplace=True)
        self.feat_fuse = nn.Sequential(    # CBR operation
            nn.Conv2d(in_c*2, in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            nn.PReLU()
        )
        self.output_cbr = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

    def forward(self, fg, bg):
        x = self.feat_fuse(torch.cat([fg, bg], dim=1))
        B, H, W, C = x.shape
        v = x
        fg = self.fg_cbr(fg)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        bg = self.bg_cbr(bg)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        attn = attn_fg + attn_bg
        x_weighted_bg = attn * v
        out = self.output_cbr(x_weighted_bg)
        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        attn = attn_layer(feature_map)
        attn = attn * self.scale 
        attn = self.attn_drop(attn)     
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2)  
        x_weighted = x_weighted.reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1
        )  
        x_weighted = F.fold(
            x_weighted, 
            output_size=(H, W), 
            kernel_size=self.kernel_size,
            padding=self.padding, 
            stride=self.stride
        )
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1)) 
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted  
