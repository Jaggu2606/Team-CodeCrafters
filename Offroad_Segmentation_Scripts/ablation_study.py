"""
Ablation Study Script for Multi-Attention U-Net
Tests contribution of each architectural component:
  1. Baseline U-Net (no attention)
  2. + ASPP only
  3. + CBAM only
  4. + Attention Gates only
  5. Full Model (all components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from train_segmentation.py
sys.path.insert(0, os.path.dirname(__file__))
from train_segmentation import (
    OffroadSegDataset, get_train_transform, get_val_transform,
    compute_class_weights, DiceFocalLoss,
    evaluate_metrics, N_CLASSES, class_names, color_palette,
    ChannelAttention, SpatialAttention, CBAM, AttentionGate, 
    ASPP, ASPPConv, ASPPPooling
)

import torchvision.models as models


# ============================================================================
# Ablation Model Variants
# ============================================================================

class BaselineUNet(nn.Module):
    """Standard U-Net with ConvNeXt-Tiny encoder (NO attention mechanisms)"""
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        convnext = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)

        self.encoder = create_feature_extractor(convnext, {
            'features.0': 'e0', 'features.1': 'e1', 'features.3': 'e2',
            'features.5': 'e3', 'features.7': 'e4',
        })

        # Simple decoder blocks (no attention gates)
        self.decoder3 = self._make_decoder_block(768, 384, 256)
        self.decoder2 = self._make_decoder_block(256, 192, 128)
        self.decoder1 = self._make_decoder_block(128, 96, 64)

        self.refine = nn.Sequential(
            nn.Conv2d(64 + 96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, 2, stride=2), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def _make_decoder_block(self, in_ch, skip_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.encoder(x)
        e0, e1, e2, e3, e4 = feats['e0'], feats['e1'], feats['e2'], feats['e3'], feats['e4']

        # Decoder without attention
        d3 = self.decoder3(torch.cat([nn.functional.interpolate(e4, size=e3.shape[2:], mode='bilinear'), e3], 1))
        d2 = self.decoder2(torch.cat([nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear'), e2], 1))
        d1 = self.decoder1(torch.cat([nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear'), e1], 1))

        d1 = self.refine(torch.cat([d1, e0], 1))
        out = self.final_upsample(d1)
        return self.final_conv(out)


class UNetWithASPP(nn.Module):
    """U-Net + ASPP bottleneck ONLY (no CBAM, no attention gates)"""
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        convnext = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)

        self.encoder = create_feature_extractor(convnext, {
            'features.0': 'e0', 'features.1': 'e1', 'features.3': 'e2',
            'features.5': 'e3', 'features.7': 'e4',
        })

        # ASPP bottleneck
        self.aspp = ASPP(in_channels=768, out_channels=384)

        # Simple decoder (no attention gates)
        self.decoder3 = self._make_decoder_block(384, 384, 256)
        self.decoder2 = self._make_decoder_block(256, 192, 128)
        self.decoder1 = self._make_decoder_block(128, 96, 64)

        self.refine = nn.Sequential(
            nn.Conv2d(64 + 96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, 2, stride=2), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def _make_decoder_block(self, in_ch, skip_ch, out_ch):
        return nn.Module()  # Placeholder, will use functional approach

    def forward(self, x):
        feats = self.encoder(x)
        e0, e1, e2, e3, e4 = feats['e0'], feats['e1'], feats['e2'], feats['e3'], feats['e4']

        # ASPP bottleneck
        b = self.aspp(e4)

        # Simple concatenation decoder
        b_up = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([b_up, e3], 1)
        d3 = nn.Conv2d(d3.size(1), 256, 3, padding=1).to(x.device)(d3)

        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d3_up, e2], 1)
        d2 = nn.Conv2d(d2.size(1), 128, 3, padding=1).to(x.device)(d2)

        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d2_up, e1], 1)
        d1 = nn.Conv2d(d1.size(1), 64, 3, padding=1).to(x.device)(d1)

        d1 = self.refine(torch.cat([d1, e0], 1))
        out = self.final_upsample(d1)
        return self.final_conv(out)


class UNetWithCBAM(nn.Module):
    """U-Net + CBAM in encoder ONLY (no ASPP, no attention gates)"""
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        convnext = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)

        self.encoder = create_feature_extractor(convnext, {
            'features.0': 'e0', 'features.1': 'e1', 'features.3': 'e2',
            'features.5': 'e3', 'features.7': 'e4',
        })

        # CBAM modules
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.cbam4 = CBAM(768)

        # Simple decoder
        self.decoder3 = BaselineUNet(n_classes)._make_decoder_block(768, 384, 256)
        self.decoder2 = BaselineUNet(n_classes)._make_decoder_block(256, 192, 128)
        self.decoder1 = BaselineUNet(n_classes)._make_decoder_block(128, 96, 64)

        self.refine = nn.Sequential(
            nn.Conv2d(64 + 96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, 2, stride=2), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        feats = self.encoder(x)
        e0, e1 = feats['e0'], feats['e1']
        e2 = self.cbam2(feats['e2'])
        e3 = self.cbam3(feats['e3'])
        e4 = self.cbam4(feats['e4'])

        # Simple decoder (no attention gates, no ASPP)
        d3 = self.decoder3(torch.cat([F.interpolate(e4, e3.shape[2:], mode='bilinear'), e3], 1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear'), e2], 1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear'), e1], 1))

        d1 = self.refine(torch.cat([d1, e0], 1))
        out = self.final_upsample(d1)
        return self.final_conv(out)


def get_model_variant(variant_name, n_classes=10):
    """Factory function to create model variants"""
    if variant_name == 'baseline_unet':
        return BaselineUNet(n_classes=n_classes, pretrained=True)
    elif variant_name == 'with_aspp':
        return UNetWithASPP(n_classes=n_classes, pretrained=True)
    elif variant_name == 'with_cbam':
        return UNetWithCBAM(n_classes=n_classes, pretrained=True)
    elif variant_name == 'full':
        # Import full model from train_segmentation
        from train_segmentation import MultiAttentionUNet
        return MultiAttentionUNet(n_classes=n_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


# ============================================================================
# Training Function for Ablation
# ============================================================================

def train_ablation_model(model, variant_name, train_loader, val_loader, device, args):
    """Quick training for ablation study"""
    print(f"\n{'='*80}")
    print(f"Training Variant: {variant_name}")
    print(f"{'='*80}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Optimizer and loss
    class_weights = compute_class_weights(train_loader.dataset, n_classes=N_CLASSES, max_samples=500)
    class_weights = class_weights.to(device)
    criterion = DiceFocalLoss(n_classes=N_CLASSES, gamma=3.0, class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_iou = 0.0
    history = {'val_iou': [], 'val_dice': [], 'val_acc': []}

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        val_iou, val_dice, val_acc = evaluate_metrics(model, val_loader, device, N_CLASSES)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"ablation_{variant_name}.pth")

        scheduler.step()

    return best_val_iou, history


# ============================================================================
# Main Ablation Study
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../Offroad_Segmentation_Training_Dataset', help='Dataset directory')
    parser.add_argument('--variant', default='all', choices=['baseline_unet', 'with_aspp', 'with_cbam', 'full', 'all'])
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for ablation')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=384)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_data = OffroadSegDataset(os.path.join(args.data_dir, 'train'), get_train_transform(args.img_size))
    val_data = OffroadSegDataset(os.path.join(args.data_dir, 'val'), get_val_transform(args.img_size))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Variants to test
    variants = ['baseline_unet', 'with_aspp', 'with_cbam', 'full'] if args.variant == 'all' else [args.variant]

    results = {}
    for variant in variants:
        model = get_model_variant(variant, n_classes=N_CLASSES).to(device)
        best_iou, history = train_ablation_model(model, variant, train_loader, val_loader, device, args)

        results[variant] = {
            'best_val_iou': best_iou,
            'final_val_dice': history['val_dice'][-1],
            'final_val_acc': history['val_acc'][-1],
            'history': history
        }

    # Save results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'Val IoU':<12} {'Val Dice':<12} {'Val Acc':<12}")
    print("-" * 80)
    for variant, res in results.items():
        print(f"{variant:<20} {res['best_val_iou']:<12.4f} {res['final_val_dice']:<12.4f} {res['final_val_acc']:<12.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
