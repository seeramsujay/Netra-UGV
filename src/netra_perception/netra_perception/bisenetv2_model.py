"""
NETRA-UGV BiSeNetV2 (Bilateral Segmentation Network v2)
========================================================
PyTorch implementation of the BiSeNetV2 architecture for off-road
real-time terrain semantic segmentation on edge devices (NVIDIA Jetson / RTX GPU).

Reference:
  - Yu et al., "BiSeNet V2: Bilateral Network with Guided Aggregation for
    Real-Time Semantic Segmentation", IJCV 2021.
  - MASTER_PROJECT_REPORT.md §3.1
  - WEIGHTS_AND_MODELS_GUIDE.md §2

4 Terrain Output Classes:
  0 - SOLID_GROUND:      Soil, dirt road, gravel, asphalt, concrete
  1 - PLIANT_VEGETATION: Grass, light brush (traversable at governed speed)
  2 - MUD_HAZARD:        Wet mud, marsh, puddles, water
  3 - RIGID_OBSTACLE:    Trees, rocks, poles, barriers, buildings, vehicles
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm2d + ReLU block."""

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """
    Detail Branch: Shallow, high-capacity path (3 stages, 1/8 spatial reduction)
    preserving rich spatial details (edges, boundaries, small obstacles).
    """

    def __init__(self):
        super().__init__()
        # Stage 1: 1/2 resolution, 64 channels
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
        )
        # Stage 2: 1/4 resolution, 64 channels
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
        )
        # Stage 3: 1/8 resolution, 128 channels
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):
    """
    Stem Block: Initial stage of Semantic Branch with two parallel downsampling
    paths to rapidly shrink spatial resolution while retaining context.
    """

    def __init__(self):
        super().__init__()
        self.conv_in = ConvBNReLU(3, 16, 3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        feat_l = self.left(feat)
        feat_r = self.right(feat)
        feat_cat = torch.cat([feat_l, feat_r], dim=1)
        return self.fuse(feat_cat)


class GELayer(nn.Module):
    """
    Gather-and-Expansion Layer (GE Layer):
    Inverted bottleneck with depthwise separable convolutions.
    """

    def __init__(self, in_chan: int, out_chan: int, exp_ratio: int = 6, stride: int = 1):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.stride = stride

        if stride == 1:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1),
                nn.Conv2d(in_chan, mid_chan, 3, stride=1, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1),
                nn.Conv2d(in_chan, mid_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return self.relu(self.conv(x) + x)
        else:
            return self.relu(self.conv(x) + self.shortcut(x))


class CEBlock(nn.Module):
    """
    Context Embedding Block (CE Block):
    Global Average Pooling + Squeeze-and-Excitation context gating.
    """

    def __init__(self, in_chan: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.conv_last = ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.gap(x)
        feat = self.conv_gap(feat)
        return self.conv_last(x * feat + x)


class SegmentBranch(nn.Module):
    """
    Semantic Branch: Deep, narrow path capturing high-level global context.
    """

    def __init__(self):
        super().__init__()
        self.S1_S2 = StemBlock()  # 1/4 resolution, 16 ch
        self.S3 = nn.Sequential(  # 1/8 resolution, 32 ch
            GELayer(16, 32, stride=2),
            GELayer(32, 32, stride=1),
        )
        self.S4 = nn.Sequential(  # 1/16 resolution, 64 ch
            GELayer(32, 64, stride=2),
            GELayer(64, 64, stride=1),
        )
        self.S5 = nn.Sequential(  # 1/32 resolution, 128 ch
            GELayer(64, 128, stride=2),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
        )
        self.S5_ce = CEBlock(128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat2 = self.S1_S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5 = self.S5(feat4)
        feat5 = self.S5_ce(feat5)
        return feat2, feat3, feat4, feat5


class BGALayer(nn.Module):
    """
    Bilateral Guided Aggregation (BGA) Layer:
    Fuses Detail Branch (fine spatial edges) and Semantic Branch (global context).
    """

    def __init__(self):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
        )
        self.conv = ConvBNReLU(128, 128, 3, stride=1, padding=1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right1 = F.interpolate(right1, size=dsize, mode="bilinear", align_corners=False)
        right2 = self.right2(x_s)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode="bilinear", align_corners=False)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    """
    Segmentation prediction head:
    ConvBNReLU + Dropout + 1x1 Conv + Bilinear upsampling to target spatial resolution.
    """

    def __init__(self, in_chan: int, mid_chan: int, num_classes: int, scale_factor: int = 8):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, num_classes, 1, 1, 0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if target_size is not None:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        elif self.scale_factor != 1:
            feat = F.interpolate(feat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return feat


class BiSeNetV2(nn.Module):
    """
    Full BiSeNetV2 Model for NETRA-UGV.

    Args:
        num_classes: Number of output semantic classes (default 4).
        use_aux: If True, returns auxiliary booster head outputs during training.
                 During eval / export, only returns the primary segmentation map.
    """

    def __init__(self, num_classes: int = 4, use_aux: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux

        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 128, num_classes, scale_factor=8)

        if use_aux:
            self.aux2 = SegmentHead(16, 64, num_classes, scale_factor=4)
            self.aux3 = SegmentHead(32, 64, num_classes, scale_factor=8)
            self.aux4 = SegmentHead(64, 128, num_classes, scale_factor=16)
            self.aux5 = SegmentHead(128, 128, num_classes, scale_factor=32)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        target_size = (x.size(2), x.size(3))
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5 = self.segment(x)
        feat_head = self.bga(feat_d, feat5)
        out = self.head(feat_head, target_size=target_size)

        if self.training and self.use_aux:
            aux2 = self.aux2(feat2, target_size=target_size)
            aux3 = self.aux3(feat3, target_size=target_size)
            aux4 = self.aux4(feat4, target_size=target_size)
            aux5 = self.aux5(feat5, target_size=target_size)
            return out, aux2, aux3, aux4, aux5

        return out
