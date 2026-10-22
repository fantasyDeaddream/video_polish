"""Inference-only Practical-RIFE 4.25 / 4.25.lite, MIT (c) 2021 hzwer.

Derived from the matching official model archives; source and license are recorded
in models/rife/manifest.json and docs/licenses/Practical-RIFE-MIT.txt.
The topology/weights are unchanged. Endpoint feature encoding is cached, and
warping uses FP32 coordinates even when convolution runs in FP16.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, 3, 2, 1), nn.LeakyReLU(.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(.2, True)

    def forward(self, x):
        return self.cnn3(self.relu(self.cnn2(self.relu(self.cnn1(self.relu(self.cnn0(x)))))))


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))
        self.relu = nn.LeakyReLU(.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2), conv(c // 2, c))
        self.convblock = nn.Sequential(*(ResConv(c) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1. / scale, mode='bilinear', align_corners=False)
        if flow is not None:
            small_flow = F.interpolate(flow, scale_factor=1. / scale, mode='bilinear', align_corners=False) / scale
            x = torch.cat((x, small_flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = F.interpolate(tmp, scale_factor=scale, mode='bilinear', align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class RifeNetwork(nn.Module):
    def __init__(self, lite=False, scale=1.):
        super().__init__()
        self.block0 = IFBlock(15, 192)
        self.block1 = IFBlock(28, 128)
        self.block2 = IFBlock(28, 96)
        self.block3 = IFBlock(28, 64)
        self.block4 = IFBlock(28, 24 if lite else 32)
        self.encode = Head()
        self.scales = [s / scale for s in ([32, 16, 8, 4, 1] if lite else [16, 8, 4, 2, 1])]
        self.register_buffer('grid', torch.empty(0), persistent=False)

    def configure_grid(self, height, width, device):
        horizontal = torch.linspace(-1, 1, width, device=device, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width)
        vertical = torch.linspace(-1, 1, height, device=device, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width)
        self.grid = torch.cat((horizontal, vertical), 1)
        self.height, self.width = height, width

    def warp(self, image, flow):
        normal = torch.cat((flow[:, :1].float() / ((self.width - 1.) / 2.),
                            flow[:, 1:2].float() / ((self.height - 1.) / 2.)), 1)
        grid = (self.grid + normal).permute(0, 2, 3, 1)
        return F.grid_sample(image.float(), grid, mode='bilinear', padding_mode='border', align_corners=True).to(image.dtype)

    def forward(self, img0, img1, timestep, f0, f1):
        timestep = timestep.expand_as(img0[:, :1])
        warped0, warped1, flow = img0, img1, None
        for block, scale in zip((self.block0, self.block1, self.block2, self.block3, self.block4), self.scales):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale)
            else:
                wf0, wf1 = self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4])
                delta, mask, feat = block(torch.cat((warped0, warped1, wf0, wf1, timestep, mask, feat), 1), flow, scale)
                flow = flow + delta
            warped0, warped1 = self.warp(img0, flow[:, :2]), self.warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)
