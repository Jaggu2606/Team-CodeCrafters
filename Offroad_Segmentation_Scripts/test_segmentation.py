"""
Multi-Attention U-Net with ASPP — Test/Evaluation Script
Offroad Semantic Segmentation (10 classes)

Evaluates trained model on test dataset with optional TTA (Test-Time Augmentation).
Saves prediction masks, colored visualizations, and per-class metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
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


# ============================================================================
# Model Components (same as training script)
# ============================================================================

class ChannelAttention(nn.Module):
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
        return torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class AttentionGate(nn.Module):
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
    """Multi-Attention U-Net with ConvNeXt-Tiny encoder."""
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

        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.cbam4 = CBAM(768)

        self.aspp = ASPP(in_channels=768, out_channels=384)

        self.decoder3 = DecoderBlock(384, 384, 256)
        self.decoder2 = DecoderBlock(256, 192, 128)
        self.decoder1 = DecoderBlock(128, 96, 64)

        self.refine = nn.Sequential(
            nn.Conv2d(64 + 96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        e0 = features['e0']
        e1 = features['e1']
        e2 = self.cbam2(features['e2'])
        e3 = self.cbam3(features['e3'])
        e4 = self.cbam4(features['e4'])

        b = self.aspp(e4)

        d3 = self.decoder3(b, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        d1 = self.refine(torch.cat([d1, e0], dim=1))

        out = self.final_upsample(d1)
        return self.final_conv(out)


class LiteDecoderBlock(nn.Module):
    """Lightweight decoder block using depthwise separable convolutions."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate = AttentionGate(g_channels=in_channels, x_channels=skip_channels)
        combined = in_channels + skip_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(combined, combined, 3, padding=1, groups=combined, bias=False),
            nn.Conv2d(combined, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
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
    """Lightweight Multi-Attention U-Net with MobileNetV3-Small encoder."""
    def __init__(self, n_classes=10, pretrained=True):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        backbone = models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        self.encoder = create_feature_extractor(backbone, {
            'features.0': 'e0', 'features.1': 'e1',
            'features.3': 'e2', 'features.8': 'e3', 'features.12': 'e4',
        })
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(24)
        self.cbam3 = CBAM(48)
        self.cbam4 = CBAM(576)
        self.aspp = ASPP(in_channels=576, out_channels=128, rates=(6, 12, 18))
        self.decoder4 = LiteDecoderBlock(128, 48, 64)
        self.decoder3 = LiteDecoderBlock(64, 24, 32)
        self.decoder2 = LiteDecoderBlock(32, 16, 24)
        self.decoder1 = LiteDecoderBlock(24, 16, 16)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        e0 = features['e0']
        e1 = self.cbam1(features['e1'])
        e2 = self.cbam2(features['e2'])
        e3 = self.cbam3(features['e3'])
        e4 = self.cbam4(features['e4'])
        b = self.aspp(e4)
        d4 = self.decoder4(b, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        out = self.final_upsample(d1)
        return self.final_conv(out)


# ============================================================================
# Dataset
# ============================================================================

class OffroadSegDataset(Dataset):
    def __init__(self, data_dir, transform=None, return_filename=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.return_filename = return_filename

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

        if self.return_filename:
            return image, new_mask.long(), data_id
        return image, new_mask.long()


def get_val_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute per-class IoU, return (mean_iou, list of per-class iou)."""
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

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute per-class Dice, return (mean_dice, list of per-class dice)."""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    dice_per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        dice = (2.0 * intersection + smooth) / (pred_c.sum().float() + target_c.sum().float() + smooth)
        dice_per_class.append(dice.item())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def mask_to_color(mask):
    """Convert class mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color_mask[mask == c] = color_palette[c]
    return color_mask


def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save side-by-side comparison of input, ground truth, and prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = np.clip(img * std + mean, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary text file and per-class bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy:    {results['pixel_acc']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results['class_iou']):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

        f.write("\nPer-Class Dice:\n")
        f.write("-" * 40 + "\n")
        for name, dice in zip(class_names, results['class_dice']):
            f.write(f"  {name:<20}: {dice:.4f}\n")

    print(f"Saved evaluation metrics to {filepath}")

    # Per-class IoU bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(N_CLASSES), valid_iou,
           color=[color_palette[i] / 255 for i in range(N_CLASSES)], edgecolor='black')
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Test-Time Augmentation
# ============================================================================

@torch.no_grad()
def predict_with_tta(model, image, device, scales=(0.75, 1.0, 1.25)):
    """
    Test-Time Augmentation: horizontal flip + multi-scale.
    image: (1, 3, H, W) normalized tensor
    Returns: (1, N_CLASSES, H, W) averaged logits
    """
    H, W = image.shape[2], image.shape[3]
    accumulated = torch.zeros(1, N_CLASSES, H, W, device=device)
    count = 0

    for scale in scales:
        sH, sW = int(H * scale), int(W * scale)
        # Ensure dimensions are even (for proper downsampling in encoder)
        sH = max(sH - sH % 32, 32)
        sW = max(sW - sW % 32, 32)
        scaled = F.interpolate(image, size=(sH, sW), mode='bilinear', align_corners=False)

        # Original
        logits = model(scaled)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        accumulated += logits
        count += 1

        # Horizontal flip
        flipped = torch.flip(scaled, dims=[3])
        logits_flip = model(flipped)
        logits_flip = torch.flip(logits_flip, dims=[3])
        logits_flip = F.interpolate(logits_flip, size=(H, W), mode='bilinear', align_corners=False)
        accumulated += logits_flip
        count += 1

    return accumulated / count


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Evaluate Multi-Attention U-Net on test data')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'best_model.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of comparison visualizations to save')
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Augmentation (slower but more accurate)')
    parser.add_argument('--lite', action='store_true',
                        help='Use lightweight MobileNetV3-Small model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.lite:
        model = MultiAttentionUNetLite(n_classes=N_CLASSES, pretrained=False)
        print("Using LITE model (MobileNetV3-Small)")
    else:
        model = MultiAttentionUNet(n_classes=N_CLASSES, pretrained=False)
        print("Using FULL model (ResNet34)")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = OffroadSegDataset(args.data_dir, transform=get_val_transform(args.img_size),
                                return_filename=True)
    print(f"Loaded {len(dataset)} samples")

    if args.tta:
        print("TTA enabled — processing one image at a time")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate
    iou_scores, dice_scores, pixel_accuracies = [], [], []
    all_class_iou, all_class_dice = [], []
    sample_count = 0

    print(f"\nProcessing {len(dataset)} images...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            if args.tta:
                outputs = predict_with_tta(model, imgs, device)
            else:
                outputs = model(imgs)

            labels_squeezed = labels.squeeze(dim=1).long() if labels.dim() == 4 else labels
            predicted_masks = torch.argmax(outputs, dim=1)

            # Metrics
            iou, class_iou = compute_iou(outputs, labels_squeezed, N_CLASSES)
            dice, class_dice = compute_dice(outputs, labels_squeezed, N_CLASSES)
            pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)

                # Raw mask (class IDs 0-9)
                cv2.imwrite(os.path.join(masks_dir, f'{base_name}_pred.png'), pred_mask)

                # Colored mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Comparison visualization (first N samples)
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count:03d}.png'),
                        data_id
                    )

                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # Aggregate results
    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pixel_acc,
        'class_iou': avg_class_iou,
        'class_dice': avg_class_dice,
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Pixel Accuracy:    {mean_pixel_acc:.4f}")
    print("=" * 50)

    print("\nPer-Class IoU:")
    for name, iou in zip(class_names, avg_class_iou):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"  {name:<20}: {iou_str}")

    # Save results
    save_metrics_summary(results, args.output_dir)

    tta_str = " (with TTA)" if args.tta else ""
    print(f"\nEvaluation complete{tta_str}! Processed {len(dataset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparisons ({min(args.num_samples, len(dataset))} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()
