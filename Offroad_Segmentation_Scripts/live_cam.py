"""
Multi-Attention U-Net with ASPP — Live Webcam Segmentation
Offroad Semantic Segmentation (10 classes)

Real-time inference on webcam feed with colored segmentation overlay.
Usage: python live_cam.py --model_path best_model_full.pth --camera_id 0
Press 'q' to quit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import os
import argparse
import time


# ============================================================================
# Constants
# ============================================================================

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

N_CLASSES = 10
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# Model Components (must match train_segmentation.py exactly)
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


# ============================================================================
# Full Model — ConvNeXt-Tiny encoder
# ============================================================================

class MultiAttentionUNet(nn.Module):
    """Multi-Attention U-Net with ASPP bottleneck and ConvNeXt-Tiny encoder."""
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


# ============================================================================
# Lite Model — MobileNetV3-Small encoder
# ============================================================================

class LiteDecoderBlock(nn.Module):
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
            'features.0': 'e0',
            'features.1': 'e1',
            'features.3': 'e2',
            'features.8': 'e3',
            'features.12': 'e4',
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
# Utilities
# ============================================================================

def mask_to_color(mask):
    """Convert class mask to RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color_mask[mask == c] = color_palette[c]
    return color_mask


def preprocess_frame(frame):
    """Convert BGR frame to normalized tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    normalized = (resized - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def draw_legend(frame, y_start=60):
    """Draw class color legend on the frame."""
    for i, name in enumerate(class_names):
        y = y_start + i * 25
        color = tuple(int(c) for c in color_palette[i][::-1])  # RGB -> BGR
        cv2.rectangle(frame, (10, y), (30, y + 18), color, -1)
        cv2.rectangle(frame, (10, y), (30, y + 18), (255, 255, 255), 1)
        cv2.putText(frame, name, (35, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Live webcam offroad segmentation')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'best_model_full.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (overrides camera)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay transparency (0=camera only, 1=mask only)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Model input resolution')
    parser.add_argument('--lite', action='store_true',
                        help='Use lightweight MobileNetV3-Small model')
    parser.add_argument('--save_output', type=str, default=None,
                        help='Save output video to this path (no GUI needed)')
    args = parser.parse_args()

    global IMG_SIZE
    IMG_SIZE = args.img_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.lite:
        model = MultiAttentionUNetLite(n_classes=N_CLASSES, pretrained=False)
        print("Using LITE model (MobileNetV3-Small)")
    else:
        model = MultiAttentionUNet(n_classes=N_CLASSES, pretrained=False)
        print("Using FULL model (ConvNeXt-Tiny)")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = f"Video: {args.video}"
    else:
        cap = cv2.VideoCapture(args.camera_id)
        source_name = f"Camera ID: {args.camera_id}"
    if not cap.isOpened():
        print(f"Error: Could not open {source_name}")
        return

    print(f"\n{source_name} opened")
    print(f"Overlay alpha: {args.alpha}")

    # Get video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else 0

    # Setup video writer if saving
    video_writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_output, fourcc, src_fps, (orig_w, orig_h))
        print(f"Saving output to: {args.save_output}")

    if not args.save_output:
        print("Press 'q' to quit, '+'/'-' to adjust overlay transparency")

    fps_smoothed = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                print(f"\nVideo processing complete! {frame_count} frames processed.")
            else:
                print("Failed to read frame from camera")
            break

        frame_count += 1
        t0 = time.time()

        # Preprocess and inference
        tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():
            logits = model(tensor)

        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Colorize and resize to original frame size
        color_mask = mask_to_color(pred)
        color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Overlay segmentation on frame
        overlay = cv2.addWeighted(frame, 1.0 - args.alpha, color_mask_bgr, args.alpha, 0)

        # FPS counter (exponential moving average)
        dt = time.time() - t0
        fps_instant = 1.0 / (dt + 1e-9)
        fps_smoothed = 0.8 * fps_smoothed + 0.2 * fps_instant

        cv2.putText(overlay, f"FPS: {fps_smoothed:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw legend
        draw_legend(overlay)

        if video_writer:
            video_writer.write(overlay)
            # Print progress
            if total_frames > 0 and frame_count % 30 == 0:
                pct = frame_count / total_frames * 100
                print(f"  Processing: {frame_count}/{total_frames} ({pct:.0f}%) - FPS: {fps_smoothed:.1f}", end='\r')
        else:
            cv2.imshow('Offroad Segmentation - Live', overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                args.alpha = min(1.0, args.alpha + 0.05)
                print(f"Alpha: {args.alpha:.2f}")
            elif key == ord('-'):
                args.alpha = max(0.0, args.alpha - 0.05)
                print(f"Alpha: {args.alpha:.2f}")

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\nOutput saved to: {args.save_output}")
    else:
        cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
