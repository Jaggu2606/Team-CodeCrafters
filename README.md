# Offroad Semantic Segmentation - Team CodeCrafters

**Real-time semantic segmentation for offroad/outdoor environments using a custom Multi-Attention U-Net architecture.**

Built for **Krackthon Hackathon** - segmenting offroad terrain into 10 semantic classes for autonomous navigation and scene understanding.

---

## Problem Statement

Autonomous vehicles operating in offroad environments need to understand the terrain around them - distinguishing between navigable ground, obstacles like rocks and logs, and vegetation. Unlike urban self-driving (roads, cars, pedestrians), offroad scenes have highly irregular textures and no lane markings.

We built a **real-time semantic segmentation system** that classifies every pixel of an offroad image into one of **10 terrain classes**.

## 10 Semantic Classes

| Class | Color | Description |
|-------|-------|-------------|
| Background | Black | Unclassified regions |
| Trees | Forest Green | Tree canopy and trunks |
| Lush Bushes | Lime | Green/healthy bushes |
| Dry Grass | Tan | Dried grass areas |
| Dry Bushes | Brown | Dead/dry bush vegetation |
| Ground Clutter | Olive | Debris, fallen leaves |
| Logs | Saddle Brown | Fallen tree logs |
| Rocks | Gray | Rock formations |
| Landscape | Sienna | Open terrain/dirt |
| Sky | Sky Blue | Sky region |

---

## Architecture: Multi-Attention U-Net with ASPP

We designed a custom **encoder-decoder segmentation network** combining multiple attention mechanisms for superior feature extraction.

### High-Level Architecture

```
Input Image (384x384)
        |
   [ConvNeXt-Tiny Encoder]  (Pretrained ImageNet)
        |
   Feature Maps at 4 scales:
     e0: 96ch  (H/4)   -- stem features
     e1: 96ch  (H/4)   -- stage 1
     e2: 192ch (H/8)   -- stage 2 + CBAM
     e3: 384ch (H/16)  -- stage 3 + CBAM
     e4: 768ch (H/32)  -- stage 4 + CBAM
        |
   [ASPP Bottleneck]  (rates: 6, 12, 18)
     768ch -> 384ch multi-scale context
        |
   [Decoder with Attention Gates]
     d3: ASPP(384) + skip(e3) -> 256ch  (H/16)
     d2: d3(256)   + skip(e2) -> 128ch  (H/8)
     d1: d2(128)   + skip(e1) -> 64ch   (H/4)
        |
   [Refinement Block]
     Concat d1 + e0 (stem skip) -> 64ch  (H/4)
        |
   [Final Upsample]  (2x ConvTranspose2d)
     64 -> 48 -> 32 -> 10 classes  (H)
        |
   Segmentation Mask (384x384x10)
```

### Key Components

#### 1. ConvNeXt-Tiny Encoder (41.7M params)
- Pretrained on ImageNet-1K for strong feature initialization
- Modern ConvNet architecture outperforming ResNets
- Hierarchical feature extraction at 4 scales (H/4 to H/32)

#### 2. CBAM (Convolutional Block Attention Module)
Applied on encoder features at stages 2, 3, and 4:
- **Channel Attention**: "What" to focus on - learns which feature channels are important
- **Spatial Attention**: "Where" to focus on - highlights important spatial locations
- Helps the model focus on relevant terrain features and ignore noise

```
Feature -> [Channel Attention] -> [Spatial Attention] -> Refined Feature
              (squeeze-excite)       (7x7 conv on
               avg+max pool)         avg+max maps)
```

#### 3. ASPP (Atrous Spatial Pyramid Pooling)
At the bottleneck between encoder and decoder:
- Parallel dilated convolutions at rates **6, 12, 18** + global average pooling
- Captures multi-scale context without losing resolution
- Critical for offroad scenes where objects appear at varying scales

```
Input (768ch)
  |---> 1x1 conv --------->|
  |---> 3x3 dilated (r=6)->|
  |---> 3x3 dilated (r=12)>|---> Concat -> 1x1 Project -> 384ch
  |---> 3x3 dilated (r=18)>|
  |---> Global AvgPool ---->|
```

#### 4. Attention Gates on Skip Connections
Standard U-Net blindly concatenates encoder features. We use **Attention Gates** that learn to filter skip connections:
- Gating signal from decoder (coarser, semantic-rich) guides which encoder features (finer, detail-rich) to pass through
- Suppresses irrelevant encoder activations
- Improves segmentation boundaries

```
Decoder (gate) ---> W_g ---> |
                              + ---> ReLU ---> Psi (sigmoid) ---> * Encoder skip
Encoder (skip) ---> W_x ---> |
```

### Lite Model Variant (3.2M params)
For edge deployment, we also trained a lightweight version:
- **MobileNetV3-Small** encoder (instead of ConvNeXt-Tiny)
- **Depthwise separable convolutions** in decoder (instead of standard conv)
- **ReLU6** activation for quantization-friendly inference
- 13x fewer parameters, runs on mobile/embedded devices

---

## Training Details

### Loss Function: Dice + Focal Cross-Entropy
- **Dice Loss**: Handles class imbalance by measuring overlap
- **Focal CE** (gamma=3.0): Down-weights easy examples, focuses learning on hard pixels
- **Class weights**: Inverse-frequency weighting capped at [0.3, 5.0]

### Data Augmentation (Albumentations)
- RandomResizedCrop (scale 0.5-1.0) - scale invariance
- HorizontalFlip, VerticalFlip, RandomRotate90
- Affine transforms (translate, scale, rotate)
- ColorJitter (brightness, contrast, saturation, hue) - lighting robustness
- CLAHE, RandomGamma - contrast adaptation
- GaussianBlur, GaussNoise - regularization
- CoarseDropout - occlusion robustness

### Optimization
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: 5-epoch linear warmup + Cosine Annealing
- **Mixed Precision**: FP16 training with gradient clipping (max_norm=1.0)
- **Weighted Sampling**: Images containing rare classes oversampled 3-11x
- **Early Stopping**: Patience=30, min_delta=0.005 on Val IoU

### Dataset
- **Training**: 2,857 images with pixel-level segmentation masks
- **Validation**: 317 images
- **Test**: 1,002 images
- Resolution: 384x384

### Results

| Model | Parameters | Val IoU | Val Dice | Val Accuracy |
|-------|-----------|---------|----------|-------------|
| **Full (ConvNeXt-Tiny)** | 41.7M | **0.4785** | **0.6447** | **83.6%** |
| Lite (MobileNetV3-Small) | 3.2M | 0.4121 | 0.5675 | 81.6% |

#### Per-Class Test IoU (Full Model)
| Class | IoU |
|-------|-----|
| Sky | 0.971 |
| Landscape | 0.559 |
| Dry Grass | 0.409 |
| Dry Bushes | 0.294 |
| Trees | 0.278 |
| Rocks | 0.078 |

---

## Project Structure

```
Offroad_Segmentation_Scripts/
    train_segmentation.py    # Training script (full + lite models)
    test_segmentation.py     # Evaluation on test set with metrics & visualizations
    live_cam.py              # Real-time webcam/video inference with overlay
    visualize.py             # Visualization utilities
    train_stats/             # Training curves and metrics
    test_results_full/       # Test predictions and comparisons (full model)
    test_results_lite/       # Test predictions and comparisons (lite model)
```

---

## Quick Start

### Requirements
```bash
pip install torch torchvision albumentations opencv-python matplotlib tqdm tensorboard
```

### Train
```bash
cd Offroad_Segmentation_Scripts

# Train full model (ConvNeXt-Tiny)
python train_segmentation.py --batch_size 4 --epochs 150 --patience 30

# Train both full + lite
python train_segmentation.py --both --batch_size 4 --epochs 150 --patience 30

# Train lite only
python train_segmentation.py --lite --batch_size 8 --epochs 150 --patience 20
```

### Test
```bash
# Evaluate full model
python test_segmentation.py --model_path best_model_full.pth --output_dir ./test_results_full --num_samples 20

# Evaluate lite model
python test_segmentation.py --model_path best_model_lite.pth --output_dir ./test_results_lite --lite --num_samples 20

# With Test-Time Augmentation (TTA)
python test_segmentation.py --model_path best_model_full.pth --tta
```

### Live Inference
```bash
# Webcam
python live_cam.py --model_path best_model_full.pth

# Video file
python live_cam.py --model_path best_model_full.pth --video path/to/video.mp4

# Save output video (no GUI required)
python live_cam.py --model_path best_model_full.pth --video input.mp4 --save_output output.mp4

# Lite model (faster)
python live_cam.py --model_path best_model_lite.pth --lite
```

**Live controls**: `q` quit | `+`/`-` adjust overlay transparency

---

## Real-Time Performance

| Model | FPS (GPU) | Resolution |
|-------|----------|-----------|
| Full (ConvNeXt-Tiny) | ~50-70 FPS | 256x256 |
| Lite (MobileNetV3-Small) | ~100+ FPS | 256x256 |

Tested on NVIDIA GPU with CUDA.

---

## Team CodeCrafters

Built with PyTorch, love, and lots of GPU hours.
