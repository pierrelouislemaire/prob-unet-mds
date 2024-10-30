import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import deterministic_unet

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    
#----------------------------------------------------------------------------
# TimeEmbedding

class TimeEmbedding(torch.nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.linear1 = Linear(1, 64)
        self.linear2 = Linear(64, out_dims)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x

#----------------------------------------------------------------------------
# Convolutional layer

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None

    def forward(self, x, stride=1, padding=0):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = F.conv2d(x, weight=w, bias=b, stride=stride, padding=padding) if w is not None else x
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Upsampling layer

class Upsample(torch.nn.Module):
    def __init__(self, channels_in, channels_out, scale=2, mode='nearest-exact', conv=True):
        super().__init__()
        self.scale = scale
        self.mode = mode
        if conv:
            self.conv = Conv2d(channels_in, channels_out, 3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        if hasattr(self, 'conv'):
            x = self.conv(x, stride=1, padding=1)
        return x
    
#----------------------------------------------------------------------------
# Downsampling layer

class Downsample(torch.nn.Module):
    def __init__(self, channels_in, channels_out, scale=2, mode='avg', conv=True):
        super().__init__()
        self.scale = scale
        self.mode = mode
        if conv:
            self.conv = Conv2d(channels_in, channels_out, 3)

    def forward(self, x):
        if self.mode == 'avg':
            x = F.avg_pool2d(x, self.scale)
        elif self.mode == 'max':
            x = F.max_pool2d(x, self.scale)
        if hasattr(self, 'conv'):
            x = self.conv(x, stride=1, padding=1)
        return x
    
#----------------------------------------------------------------------------
# Residual block

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_conv = True):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.conv2 = Conv2d(out_channels, out_channels, 3)
        self.norm1 = GroupNorm(in_channels)
        self.norm2 = GroupNorm(out_channels)
        if skip_conv:
            self.conv3 = Conv2d(in_channels, out_channels, 3)
            self.norm3 = GroupNorm(in_channels)

    def forward(self, x):
        orig = x
        x = self.conv1(F.silu(self.norm1(x)), stride=1, padding=1)
        x = self.conv2(F.silu(self.norm2(x)), stride=1, padding=1)
        if hasattr(self, 'conv3'):
            orig = self.conv3(F.silu(self.norm3(orig)), stride=1, padding=1)
        x = x + orig
        return x
    
#----------------------------------------------------------------------------
# UNet class to be used for the downscaling network

class UNet(torch.nn.Module):
    def __init__(self, input_resolution, in_channels, base_channels, num_res_blocks, channels_mult):
        super().__init__()
        self.in_conv = Conv2d(in_channels, base_channels, 3)
        self.encoder = nn.ModuleList()
        channels = base_channels
        for level in channels_mult:
            c_in = channels
            c_out = channels = channels * 2
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(c_in, c_in))
            self.encoder.append(Downsample(c_in, c_out))
        self.middle = nn.Sequential(*[ResBlock(channels, channels) for _ in range(num_res_blocks)])
        self.time_embedding = TimeEmbedding((input_resolution//(2**len(channels_mult)))**2)
        self.decoder = nn.ModuleList()
        for level in reversed(channels_mult):
            c_in = channels
            c_out = channels = channels // 2
            self.decoder.append(Upsample(c_in, c_out))
            for _ in range(num_res_blocks):
                self.decoder.append(ResBlock(c_out, c_out))

    def forward(self, x, t):
        x = self.in_conv(x, stride=1, padding=1)
        encoder_outs = []
        for module in self.encoder:
            if isinstance(module, Downsample):
                encoder_outs.append(x)
            x = module(x)
        t = self.time_embedding(t).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x + t
        x = self.middle(x)
        for module in self.decoder:
            x = module(x)
            if isinstance(module, Upsample):
                x = x + encoder_outs.pop()
        return x
    
#----------------------------------------------------------------------------
# Downscaling model that combines UNet with the downscaling head

class LR_to_HR_UNet_v1(torch.nn.Module):
    def __init__(self, input_resolution, in_channels, ds_scale, num_res_blocks, channels_mult, out_channels):
        super().__init__()
        ds_head_channels_mult = np.arange(1, np.log2(ds_scale)+1)
        base_channels = 16 * 2 ** (len(ds_head_channels_mult) - 1)
        channels = base_channels
        self.unet = UNet(input_resolution, in_channels, base_channels, num_res_blocks, channels_mult)
        self.head_ds = nn.ModuleList()
        for level in ds_head_channels_mult:
            c_in = channels
            c_out = channels = channels // 2
            self.head_ds.append(Upsample(c_in, c_out))
            self.head_ds.append(Upsample(in_channels, c_out, scale=2**level))
            for _ in range(num_res_blocks):
                self.head_ds.append(ResBlock(c_out, c_out))
        self.out_conv = Conv2d(channels, out_channels, 3)

    def forward(self, x, t):
        orig = x
        x = self.unet(x, t)
        skip = False
        for module in self.head_ds:
            if isinstance(module, Upsample) and skip:
                s = module(orig)
                x = x + s
                skip = False
            elif isinstance(module, Upsample):
                skip = True
                x = module(x)
            else:
                x = module(x)
        x = self.out_conv(x, stride=1, padding=1)
        return x
    
class LR_to_HR_UNet_v2(torch.nn.Module):
    def __init__(self, input_resolution, in_channels, ds_scale, num_res_blocks, channels_mult, out_channels):
        super().__init__()
        ds_head_channels_mult = np.arange(1, np.log2(ds_scale)+1)
        base_channels = 16 * 2 ** (len(ds_head_channels_mult) - 1)
        channels = base_channels
        self.unet = deterministic_unet.UNet(img_resolution=(input_resolution, input_resolution), in_channels=in_channels, 
                                            out_channels=base_channels, num_blocks=num_res_blocks, channel_mult=channels_mult)
        self.head_ds = nn.ModuleList()
        for level in ds_head_channels_mult:
            c_in = channels
            c_out = channels = channels // 2
            self.head_ds.append(Upsample(c_in, c_out))
            self.head_ds.append(Upsample(in_channels, c_out, scale=2**level))
            for _ in range(num_res_blocks):
                self.head_ds.append(ResBlock(c_out, c_out))
        self.out_conv = Conv2d(channels, out_channels, 3)

    def forward(self, x, t):
        orig = x
        x = self.unet(x, t)
        skip = False
        for module in self.head_ds:
            if isinstance(module, Upsample) and skip:
                s = module(orig)
                x = x + s
                skip = False
            elif isinstance(module, Upsample):
                skip = True
                x = module(x)
            else:
                x = module(x)
        x = self.out_conv(x, stride=1, padding=1)
        return x
    
if __name__ == '__main__':
    model = LR_to_HR_UNet(3, 10, 2, [1, 2, 3, 4], 3)
    x = torch.randn(64, 3, 64, 64)
    y = model(x)
    print(y.shape)