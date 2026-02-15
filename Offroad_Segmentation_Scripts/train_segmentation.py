"""
Multi-Attention U-Net with ASPP — Training Script
Offroad Semantic Segmentation (10 classes)

Architecture:
  - Encoder: ConvNeXt-Tiny (pretrained ImageNet) / MobileNetV3-Small (lite)
  - Attention: CBAM on encoder features + Attention Gates on skip connections
  - Bottleneck: ASPP (Atrous Spatial Pyramid Pooling, rates 6/12/18)
  - Decoder: 4 upsampling stages with gated skip connections
  - Loss: Dice + Focal Cross-Entropy (with capped class weights)
  - Optimizer: AdamW + Warmup + CosineAnnealingLR
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise ImportError("albumentations is required. Install with: pip install albumentations")


# ============================================================================
# Constants
# ============================================================================

value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9,    # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

N_CLASSES = len(value_map)


def mask_to_color(mask):
    """Convert class mask (H,W) to RGB color image (H,W,3)."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color_mask[mask == c] = color_palette[c]
    return color_mask


# ============================================================================
# Model Components
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        avg_out = self.mlp(x.mean(dim=[2, 3]))
        max_out = self.mlp(x.amax(dim=[2, 3]))
        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise statistics."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel then spatial."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for skip connections.
    g: gating signal from decoder (coarser)
    x: skip connection from encoder (finer)
    """
    def __init__(self, g_channels, x_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = x_channels // 2
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        combined = self.relu(g1 + x1)
        attn = self.psi(combined)
        return x * attn


class ASPPConv(nn.Module):
    """Single ASPP branch: dilated conv + BN + ReLU."""
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    """ASPP global average pooling branch."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        out = self.relu(self.conv(self.pool(x)))
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_channels=512, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.branches = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(out)


class DecoderBlock(nn.Module):
    """Decoder block: upsample + attention-gated skip concat + 2x Conv-BN-ReLU."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.att_gate = AttentionGate(g_channels=in_channels, x_channels=skip_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.att_gate(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class MultiAttentionUNet(nn.Module):
    """Multi-Attention U-Net with ASPP bottleneck and ConvNeXt-Tiny encoder.

    ConvNeXt-Tiny feature map sizes (for 384x384 input):
      features.0 (stem)    -> 96ch,  H/4  (96x96)
      features.1 (stage 1) -> 96ch,  H/4  (96x96)
      features.3 (stage 2) -> 192ch, H/8  (48x48)
      features.5 (stage 3) -> 384ch, H/16 (24x24)
      features.7 (stage 4) -> 768ch, H/32 (12x12)

    Decoder: 3 upsample stages (H/32 -> H/16 -> H/8 -> H/4), then
    a refinement block with skip from stem, followed by 4x upsample to H.
    """
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        convnext = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)

        self.encoder = create_feature_extractor(convnext, {
            'features.0': 'e0',   # 96,  H/4
            'features.1': 'e1',   # 96,  H/4
            'features.3': 'e2',   # 192, H/8
            'features.5': 'e3',   # 384, H/16
            'features.7': 'e4',   # 768, H/32
        })

        # CBAM on encoder outputs
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.cbam4 = CBAM(768)

        # Bottleneck ASPP
        self.aspp = ASPP(in_channels=768, out_channels=384)

        # Decoder — 3 stages with upsampling skip connections
        self.decoder3 = DecoderBlock(384, 384, 256)   # ASPP(384) + e3(384) -> 256, H/16
        self.decoder2 = DecoderBlock(256, 192, 128)   # d3(256) + e2(192)  -> 128, H/8
        self.decoder1 = DecoderBlock(128, 96, 64)     # d2(128) + e1(96)   -> 64,  H/4

        # Refinement: concat with stem features at H/4, then refine
        self.refine = nn.Sequential(
            nn.Conv2d(64 + 96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final classification — upsample 4x from H/4 to H
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2),  # H/4 -> H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2),  # H/2 -> H
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        e0 = features['e0']                         # (B, 96, H/4, W/4)
        e1 = features['e1']                         # (B, 96, H/4, W/4)
        e2 = self.cbam2(features['e2'])              # (B, 192, H/8, W/8)
        e3 = self.cbam3(features['e3'])              # (B, 384, H/16, W/16)
        e4 = self.cbam4(features['e4'])              # (B, 768, H/32, W/32)

        # Bottleneck
        b = self.aspp(e4)                            # (B, 384, H/32, W/32)

        # Decoder
        d3 = self.decoder3(b, e3)                    # (B, 256, H/16, W/16)
        d2 = self.decoder2(d3, e2)                   # (B, 128, H/8, W/8)
        d1 = self.decoder1(d2, e1)                   # (B, 64, H/4, W/4)

        # Refinement with stem skip
        d1 = self.refine(torch.cat([d1, e0], dim=1)) # (B, 64, H/4, W/4)

        # Final — upsample from H/4 to H
        out = self.final_upsample(d1)                # (B, 32, H, W)
        return self.final_conv(out)                  # (B, n_classes, H, W)


class LiteDecoderBlock(nn.Module):
    """Lightweight decoder block using depthwise separable convolutions."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate = AttentionGate(g_channels=in_channels, x_channels=skip_channels)
        combined = in_channels + skip_channels
        self.conv_block = nn.Sequential(
            # Depthwise separable conv 1
            nn.Conv2d(combined, combined, 3, padding=1, groups=combined, bias=False),
            nn.Conv2d(combined, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # Depthwise separable conv 2
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.att_gate(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class MultiAttentionUNetLite(nn.Module):
    """Lightweight Multi-Attention U-Net with MobileNetV3-Small encoder (~3.5M params).
    Suitable for edge devices, real-time inference, and ONNX/TensorRT export.
    """
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        backbone = models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        # MobileNetV3-Small feature extraction points:
        #   features.0  -> 16ch, H/2  (128x128)
        #   features.1  -> 16ch, H/4  (64x64)
        #   features.3  -> 24ch, H/8  (32x32)
        #   features.8  -> 48ch, H/16 (16x16)
        #   features.12 -> 576ch, H/32 (8x8)
        self.encoder = create_feature_extractor(backbone, {
            'features.0': 'e0',   # 16,  128x128
            'features.1': 'e1',   # 16,  64x64
            'features.3': 'e2',   # 24,  32x32
            'features.8': 'e3',   # 48,  16x16
            'features.12': 'e4',  # 576, 8x8
        })

        # CBAM on encoder outputs
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(24)
        self.cbam3 = CBAM(48)
        self.cbam4 = CBAM(576)

        # Lightweight ASPP with reduced channels
        self.aspp = ASPP(in_channels=576, out_channels=128, rates=(6, 12, 18))

        # Decoder with depthwise separable convolutions
        self.decoder4 = LiteDecoderBlock(128, 48, 64)    # ASPP(128) + e3(48)  -> 64
        self.decoder3 = LiteDecoderBlock(64, 24, 32)     # d4(64) + e2(24)    -> 32
        self.decoder2 = LiteDecoderBlock(32, 16, 24)     # d3(32) + e1(16)    -> 24
        self.decoder1 = LiteDecoderBlock(24, 16, 16)     # d2(24) + e0(16)    -> 16

        # Final classification
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        e0 = features['e0']                         # (B, 16, 128, 128)
        e1 = self.cbam1(features['e1'])              # (B, 16, 64, 64)
        e2 = self.cbam2(features['e2'])              # (B, 24, 32, 32)
        e3 = self.cbam3(features['e3'])              # (B, 48, 16, 16)
        e4 = self.cbam4(features['e4'])              # (B, 576, 8, 8)

        # Bottleneck
        b = self.aspp(e4)                            # (B, 128, 8, 8)

        # Decoder
        d4 = self.decoder4(b, e3)                    # (B, 64, 16, 16)
        d3 = self.decoder3(d4, e2)                   # (B, 32, 32, 32)
        d2 = self.decoder2(d3, e1)                   # (B, 24, 64, 64)
        d1 = self.decoder1(d2, e0)                   # (B, 16, 128, 128)

        # Final
        out = self.final_upsample(d1)                # (B, 16, 256, 256)
        return self.final_conv(out)                  # (B, n_classes, 256, 256)


# ============================================================================
# Loss Function
# ============================================================================

class DiceFocalLoss(nn.Module):
    """Combined Dice Loss + Focal Cross-Entropy Loss with optional class weights."""
    def __init__(self, n_classes=10, gamma=2.0, dice_weight=1.0, focal_weight=1.0, smooth=1e-6, class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        # class_weights: tensor of shape (n_classes,) — higher weight = more emphasis
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.register_buffer('class_weights', torch.ones(n_classes))

    def dice_loss(self, pred_softmax, target):
        # target: (B, H, W) long
        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        dims = (0, 2, 3)
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        cardinality = pred_softmax.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Weighted average instead of simple mean
        weighted_dice = (dice_per_class * self.class_weights).sum() / self.class_weights.sum()
        return 1.0 - weighted_dice

    def focal_loss(self, pred_logits, target):
        # Force float32 to prevent NaN overflow with mixed precision
        pred_logits = pred_logits.float()
        ce = F.cross_entropy(pred_logits, target, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(self, pred_logits, target):
        # Force float32 for entire loss computation
        pred_logits = pred_logits.float()
        pred_softmax = F.softmax(pred_logits, dim=1)
        d_loss = self.dice_loss(pred_softmax, target)
        f_loss = self.focal_loss(pred_logits, target)
        return self.dice_weight * d_loss + self.focal_weight * f_loss


# ============================================================================
# Dataset
# ============================================================================

class OffroadSegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = cv2.imread(os.path.join(self.image_dir, data_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.masks_dir, data_id), cv2.IMREAD_UNCHANGED)
        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for raw_val, class_id in value_map.items():
            new_mask[mask == raw_val] = class_id

        if self.transform:
            augmented = self.transform(image=image, mask=new_mask)
            image = augmented['image']
            new_mask = augmented['mask']

        return image, new_mask.long()


def get_train_transform(size=256):
    return A.Compose([
        # Geometric — stronger scale/crop for better generalization
        A.RandomResizedCrop((size, size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.7),
        A.Resize(size, size),  # fallback if RandomResizedCrop didn't fire
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.25),
        A.Affine(translate_percent=0.1, scale=(0.75, 1.25), rotate=(-25, 25), p=0.5,
                 mode=cv2.BORDER_CONSTANT, cval=0),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.15),
        # Color — aggressive to handle domain shift
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.7),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.5),
        # Noise/blur — regularization
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=size // 8, max_width=size // 8,
                        fill_value=0, mask_fill_value=0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weights(dataset, n_classes=10, max_samples=500):
    """Compute inverse-frequency class weights from dataset samples."""
    print("Computing class weights from training data...")
    pixel_counts = np.zeros(n_classes, dtype=np.float64)
    n = min(len(dataset), max_samples)
    for i in tqdm(range(n), desc="Scanning masks", leave=False):
        _, mask = dataset[i]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        for c in range(n_classes):
            pixel_counts[c] += (mask_np == c).sum()

    # Inverse frequency with sqrt dampening + cap to avoid NaN from extreme weights
    total = pixel_counts.sum()
    freq = pixel_counts / total
    freq = np.clip(freq, 1e-4, None)  # floor at 0.01% to avoid extreme weights
    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.sum() * n_classes  # normalize so mean weight = 1.0
    weights = np.clip(weights, 0.3, 5.0)  # cap to [0.3, 5.0] — higher cap for rare classes

    print("Class weights:")
    for i, (name, w) in enumerate(zip(class_names, weights)):
        pct = freq[i] * 100
        print(f"  {name:18s}: weight={w:.3f}  (freq={pct:.2f}%)")
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(dataset, n_classes=10, max_samples=None):
    """Compute per-sample weights so images with rare classes are sampled more often."""
    print("Computing per-sample weights for weighted sampling...")
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    # Rare class IDs: Background(0), Ground Clutter(5), Logs(6), Lush Bushes(2), Rocks(7)
    rare_classes = {0, 2, 5, 6, 7}
    sample_weights = np.ones(len(dataset), dtype=np.float64)
    for i in tqdm(range(n), desc="Computing sample weights", leave=False):
        _, mask = dataset[i]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        unique_classes = set(np.unique(mask_np).tolist())
        # Boost weight if image contains rare classes
        rare_count = len(unique_classes & rare_classes)
        if rare_count > 0:
            sample_weights[i] = 1.0 + rare_count * 2.0  # 3x-11x boost
    print(f"  Boosted {(sample_weights > 1.0).sum()}/{n} samples containing rare classes")
    return sample_weights


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute per-class IoU and return mean IoU."""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    iou_per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute per-class Dice and return mean Dice."""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    dice_per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        dice = (2.0 * intersection + smooth) / (pred_c.sum().float() + target_c.sum().float() + smooth)
        dice_per_class.append(dice.item())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().item()


@torch.no_grad()
def evaluate_metrics(model, data_loader, device, num_classes=10):
    """Evaluate IoU, Dice, and pixel accuracy on a data loader."""
    model.eval()
    iou_scores, dice_scores, acc_scores = [], [], []

    for imgs, masks in data_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        iou_scores.append(compute_iou(outputs, masks, num_classes))
        dice_scores.append(compute_dice(outputs, masks, num_classes))
        acc_scores.append(compute_pixel_accuracy(outputs, masks))

    return np.nanmean(iou_scores), np.mean(dice_scores), np.mean(acc_scores)


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    """Save training metric plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train')
    axes[0, 1].plot(history['val_iou'], label='Val')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_dice'], label='Train')
    axes[1, 0].plot(history['val_dice'], label='Val')
    axes[1, 0].set_title('Mean Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_pixel_acc'], label='Train')
    axes[1, 1].plot(history['val_pixel_acc'], label='Val')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i], history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Training Function
# ============================================================================

def train_model(model, model_tag, train_loader, val_loader, args, device, script_dir, output_dir, class_weights=None):
    """Train a single model variant. Returns best val IoU."""
    from torchvision.utils import make_grid

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # TensorBoard — separate run per model variant
    log_dir = os.path.join(script_dir, 'runs', model_tag)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Loss, optimizer, scheduler (warmup + cosine annealing)
    criterion = DiceFocalLoss(n_classes=N_CLASSES, gamma=3.0, class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # History
    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_iou', 'val_iou',
        'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc'
    ]}

    best_val_iou = 0.0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()

            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            if not math.isnan(loss_val):
                train_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                # Compute loss in float32 to avoid nan from float16 overflow
                loss = criterion(outputs.float(), masks)
                loss_val = loss.item()
                if not math.isnan(loss_val):
                    val_losses.append(loss_val)

        # --- NaN guard: if all train losses were NaN, reload best model and cut LR ---
        if len(train_losses) == 0:
            print(f"  !! All train losses were NaN at epoch {epoch+1}. Reloading best model and halving LR.")
            best_path = os.path.join(script_dir, f'best_model_{model_tag}.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
                print(f"     Reloaded best model. New LR: {optimizer.param_groups[0]['lr']:.2e}")
            continue

        # --- Metrics ---
        train_iou, train_dice, train_acc = evaluate_metrics(model, train_loader, device, N_CLASSES)
        val_iou, val_dice, val_acc = evaluate_metrics(model, val_loader, device, N_CLASSES)

        # Store
        epoch_train_loss = np.mean(train_losses) if train_losses else float('nan')
        epoch_val_loss = np.mean(val_losses) if val_losses else float('nan')
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_acc)
        history['val_pixel_acc'].append(val_acc)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {lr_now:.2e}")

        # --- TensorBoard logging ---
        writer.add_scalars('Loss', {'Train': epoch_train_loss, 'Val': epoch_val_loss}, epoch + 1)
        writer.add_scalars('IoU', {'Train': train_iou, 'Val': val_iou}, epoch + 1)
        writer.add_scalars('Dice', {'Train': train_dice, 'Val': val_dice}, epoch + 1)
        writer.add_scalars('Pixel_Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch + 1)
        writer.add_scalar('LR', lr_now, epoch + 1)

        # Log sample prediction images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_imgs, sample_masks = next(iter(val_loader))
                sample_imgs = sample_imgs.to(device)
                sample_preds = torch.argmax(model(sample_imgs), dim=1)

                n_vis = min(4, sample_imgs.shape[0])
                for i in range(n_vis):
                    img_vis = sample_imgs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_vis = torch.clamp(img_vis * std + mean, 0, 1)

                    gt_color = torch.from_numpy(
                        mask_to_color(sample_masks[i].numpy().astype(np.uint8))
                    ).permute(2, 0, 1).float() / 255.0
                    pred_color = torch.from_numpy(
                        mask_to_color(sample_preds[i].cpu().numpy().astype(np.uint8))
                    ).permute(2, 0, 1).float() / 255.0

                    grid = torch.stack([img_vis, gt_color, pred_color])
                    grid_img = make_grid(grid, nrow=3, padding=4, pad_value=1.0)
                    writer.add_image(f'Predictions/sample_{i}', grid_img, epoch + 1)

        # Early stopping + checkpoint (requires min_delta improvement)
        if val_iou > best_val_iou + args.min_delta:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(script_dir, f'best_model_{model_tag}.pth'))
            print(f"  -> New best model saved (Val IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            # Still save if it's the absolute best (even below min_delta)
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), os.path.join(script_dir, f'best_model_{model_tag}.pth'))
                print(f"  -> Model saved (Val IoU: {val_iou:.4f}, below min_delta, patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement > {args.min_delta} for {args.patience} epochs)")
                break

    # Save final model, plots, and history
    tag_output_dir = os.path.join(output_dir, model_tag)
    torch.save(model.state_dict(), os.path.join(script_dir, f'final_model_{model_tag}.pth'))
    print(f"\nSaved final model to '{script_dir}/final_model_{model_tag}.pth'")

    save_training_plots(history, tag_output_dir)
    save_history_to_file(history, tag_output_dir)

    writer.close()

    print("\n" + "=" * 50)
    print(f"TRAINING COMPLETE [{model_tag.upper()}]")
    print("=" * 50)
    print(f"  Best Val IoU:      {best_val_iou:.4f}")
    print(f"  Final Val Loss:    {history['val_loss'][-1]:.4f}")
    print(f"  Final Val Dice:    {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Acc:     {history['val_pixel_acc'][-1]:.4f}")

    return best_val_iou


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Attention U-Net with ASPP')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.005, help='Minimum IoU improvement to reset early stopping')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from (best_model.pth / final_model.pth)")


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lite', action='store_true',
                       help='Train ONLY the lightweight model (MobileNetV3-Small, ~3.2M params)')
    group.add_argument('--both', action='store_true',
                       help='Train BOTH full (ConvNeXt-Tiny) and lite (MobileNetV3) models sequentially')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Datasets (shared across both models)
    trainset = OffroadSegDataset(train_dir, transform=get_train_transform(args.img_size))
    valset = OffroadSegDataset(val_dir, transform=get_val_transform(args.img_size))

    # Weighted sampler — oversample images containing rare classes
    sample_weights = compute_sample_weights(trainset, n_classes=N_CLASSES)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(trainset), replacement=True)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    print(f"TensorBoard: tensorboard --logdir {os.path.join(script_dir, 'runs')}\n")

    # Determine which models to train
    if args.both:
        variants = [
            ('full', MultiAttentionUNet(n_classes=N_CLASSES, pretrained=True)),
            ('lite', MultiAttentionUNetLite(n_classes=N_CLASSES, pretrained=True)),
        ]
    elif args.lite:
        variants = [('lite', MultiAttentionUNetLite(n_classes=N_CLASSES, pretrained=True))]
    else:
        variants = [('full', MultiAttentionUNet(n_classes=N_CLASSES, pretrained=True))]

    # Compute class weights from training data
    class_weights = compute_class_weights(trainset, n_classes=N_CLASSES).to(device)

    results = {}
    for idx, (tag, model) in enumerate(variants):
        if len(variants) > 1:
            print(f"\n{'#' * 80}")
            print(f"# TRAINING MODEL {idx+1}/{len(variants)}: {tag.upper()}")
            print(f"{'#' * 80}")

        model = model.to(device)
        best_iou = train_model(model, tag, train_loader, val_loader, args, device, script_dir, output_dir, class_weights=class_weights)
        results[tag] = best_iou

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Summary
    if len(results) > 1:
        print(f"\n{'=' * 50}")
        print("ALL TRAINING COMPLETE")
        print(f"{'=' * 50}")
        for tag, iou in results.items():
            print(f"  {tag.upper():5s} model -> Best Val IoU: {iou:.4f} | Saved: best_model_{tag}.pth")


if __name__ == "__main__":
    main()
