"""
SGGCMNet: Spectral Group-Wise Gated CNN-Mamba Network
with Cross-Stage Mutual Distillation for HSI Classification.

Stem(1x1+3x3 Conv) -> CNNMambaFusionBlock x2, each with a
classification head. Three heads (cls0, cls1, cls2) are supervised
via progressive deep supervision with temperature-regulated
cross-stage mutual distillation (TCMD) and uncertainty-based
dynamic loss weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class StemBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, group_num=4):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False)
        self.norm = nn.GroupNorm(group_num, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpectralGroupGating(nn.Module):
    def __init__(self, channels, num_heads=4, reduction=4):
        super(SpectralGroupGating, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.head_attns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 2, self.head_dim // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(self.head_dim // reduction, self.head_dim * 2),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])

        self.head_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=num_heads, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, cnn_feat, mamba_feat):
        B, C, H, W = cnn_feat.shape
        cnn_heads = cnn_feat.view(B, self.num_heads, self.head_dim, H, W)
        mamba_heads = mamba_feat.view(B, self.num_heads, self.head_dim, H, W)

        fused_heads = []
        for i in range(self.num_heads):
            cnn_h = cnn_heads[:, i]
            mamba_h = mamba_heads[:, i]
            cnn_pool = self.avg_pool(cnn_h).view(B, self.head_dim)
            mamba_pool = self.avg_pool(mamba_h).view(B, self.head_dim)
            combined = torch.cat([cnn_pool, mamba_pool], dim=1)
            weights = self.head_attns[i](combined).view(B, 2, self.head_dim, 1, 1)
            fused_h = weights[:, 0] * cnn_h + weights[:, 1] * mamba_h
            fused_heads.append(fused_h)

        return self.head_fusion(torch.cat(fused_heads, dim=1))


class CNNMambaFusionBlock(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, num_heads=4):
        super(CNNMambaFusionBlock, self).__init__()
        self.use_residual = use_residual

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2, use_fast_path=False)
        self.mamba_norm = nn.GroupNorm(group_num, channels)
        self.mamba_act = nn.SiLU()

        self.fusion = SpectralGroupGating(channels, num_heads=num_heads)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        cnn_feat = self.cnn_branch(x)

        x_re = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_re.view(B, H * W, C)
        x_flat_fp32 = x_flat.float()
        x_mamba = self.mamba(x_flat_fp32)
        x_mamba = x_mamba.to(x.dtype)
        x_mamba = x_mamba.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        mamba_feat = self.mamba_act(self.mamba_norm(x_mamba))

        fused = self.fusion(cnn_feat, mamba_feat)

        if self.use_residual:
            return fused + identity
        return fused


class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total

    def get_weights(self):
        return [torch.exp(-v).item() for v in self.log_vars]


class SGGCMNet(nn.Module):
    """
    SGGCMNet with progressive deep supervision and cross-stage mutual distillation.

    Stem -> cls_head0, FusionBlock1 -> cls_head1, FusionBlock2 -> cls_head2.
    Each stage supervised by CE loss, with TCMD-based bidirectional
    knowledge transfer across stages.
    """
    def __init__(self, in_channels, num_classes, hidden_dim=64, group_num=4, num_heads=4, **kwargs):
        super(SGGCMNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.stem = StemBlock(in_channels, hidden_dim, group_num)
        self.fusion_block1 = CNNMambaFusionBlock(hidden_dim, True, group_num, num_heads=num_heads)
        self.fusion_block2 = CNNMambaFusionBlock(hidden_dim, True, group_num, num_heads=num_heads)

        def make_cls_head():
            return nn.Sequential(
                nn.Conv2d(hidden_dim, 128, 1, 1, 0),
                nn.GroupNorm(group_num, 128),
                nn.SiLU(),
                nn.Conv2d(128, num_classes, 1, 1, 0)
            )

        self.cls_head0 = make_cls_head()
        self.cls_head1 = make_cls_head()
        self.cls_head2 = make_cls_head()

    def forward(self, x):
        feat0 = self.stem(x)
        feat1 = self.fusion_block1(feat0)
        feat2 = self.fusion_block2(feat1)

        logits0 = self.cls_head0(feat0)
        logits1 = self.cls_head1(feat1)
        logits2 = self.cls_head2(feat2)

        return logits0, logits1, logits2
