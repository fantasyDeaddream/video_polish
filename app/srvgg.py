"""Real-ESRGAN SRVGGNetCompact; upstream BSD-3-Clause, Xintao Wang (2021).

Architecture and state-dict names match the official implementation:
https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
The BasicSR registry dependency is omitted.
"""
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        def activation():
            if act_type == 'relu':
                return nn.ReLU(inplace=True)
            if act_type == 'prelu':
                return nn.PReLU(num_parameters=num_feat)
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
