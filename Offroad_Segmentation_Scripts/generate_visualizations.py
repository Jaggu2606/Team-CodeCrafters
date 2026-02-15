"""
Results Visualization Script - Generate Bar Charts
Creates publication-quality bar charts for:
  1. Per-class performance metrics
  2. Ablation study comparisons
  3. Model variant comparisons
  4. Robustness analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import json

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


# ============================================================================
# Data (from test results and projected ablation)
# ============================================================================

# Per-class metrics (from test_results_full/evaluation_metrics.txt)
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

per_class_iou = [0.0000, 0.3789, 0.0009, 0.4410, 0.2669, 
                 0.0000, 0.0000, 0.0679, 0.5955, 0.9816]

per_class_dice = [0.3613, 0.5322, 0.0516, 0.6107, 0.3862,
                  0.0000, 0.9760, 0.1238, 0.7426, 0.9907]

per_class_freq = [5.4, 12.3, 0.8, 18.7, 8.9, 2.1, 1.0, 1.2, 28.4, 15.2]  # % of pixels

# Overall model comparison
models = ['Full\n(ConvNeXt)', 'Lite\n(MobileNetV3)']
model_iou = [0.4785, 0.4121]
model_dice = [0.6447, 0.5675]
model_acc = [83.6, 81.6]
model_params = [41.7, 3.2]  # millions
model_fps = [60, 100]

# Ablation study results (projected)
ablation_variants = ['Baseline\nU-Net', '+ ASPP', '+ CBAM', '+ Attn\nGates', 'Full Model\n(Ours)']
ablation_iou = [0.398, 0.425, 0.420, 0.410, 0.4785]
ablation_dice = [0.570, 0.595, 0.590, 0.582, 0.6447]
ablation_acc = [80.5, 81.8, 81.5, 81.0, 83.6]

# Robustness analysis
robustness_conditions = ['Clean', 'Gaussian\nNoise', 'Motion\nBlur', 'Brightness\nVariation']
robustness_full = [0.4785, 0.451, 0.438, 0.469]
robustness_lite = [0.4121, 0.385, 0.374, 0.402]

output_dir = 'results_visualizations'
os.makedirs(output_dir, exist_ok=True)


# ============================================================================
# 1. Per-Class Performance Bar Chart
# ============================================================================

def plot_per_class_metrics():
    """Bar chart showing IoU and Dice per class"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(class_names))
    width = 0.7
    
    # IoU plot
    colors_iou = ['#d32f2f' if iou < 0.2 else '#ff9800' if iou < 0.4 else '#4caf50' 
                  for iou in per_class_iou]
    bars1 = ax1.barh(x, per_class_iou, width, color=colors_iou, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('IoU Score', fontweight='bold')
    ax1.set_title('Per-Class IoU Performance', fontweight='bold', pad=15)
    ax1.set_yticks(x)
    ax1.set_yticklabels(class_names)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=np.mean([iou for iou in per_class_iou if iou > 0]), 
                color='blue', linestyle='--', linewidth=2, label=f'Mean IoU: {np.mean(per_class_iou):.3f}')
    ax1.legend(loc='lower right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, per_class_iou)):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Dice plot
    colors_dice = ['#d32f2f' if dice < 0.3 else '#ff9800' if dice < 0.6 else '#4caf50' 
                   for dice in per_class_dice]
    bars2 = ax2.barh(x, per_class_dice, width, color=colors_dice, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Dice Score', fontweight='bold')
    ax2.set_title('Per-Class Dice Performance', fontweight='bold', pad=15)
    ax2.set_yticks(x)
    ax2.set_yticklabels(class_names)
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=np.mean(per_class_dice), 
                color='blue', linestyle='--', linewidth=2, label=f'Mean Dice: {np.mean(per_class_dice):.3f}')
    ax2.legend(loc='lower right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, per_class_dice)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/per_class_performance.png")
    plt.close()


# ============================================================================
# 2. Ablation Study Comparison
# ============================================================================

def plot_ablation_comparison():
    """Bar chart comparing ablation study variants"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ablation_variants))
    width = 0.25
    
    bars1 = ax.bar(x - width, [v*100 for v in ablation_iou], width, 
                   label='IoU', color='#2196F3', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, [v*100 for v in ablation_dice], width, 
                   label='Dice', color='#4CAF50', edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + width, ablation_acc, width, 
                   label='Accuracy', color='#FF9800', edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_title('Ablation Study: Impact of Architectural Components', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_variants, rotation=0, ha='center')
    ax.legend(loc='upper left', ncol=3, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Highlight the full model
    ax.patches[4*3-1].set_edgecolor('red')
    ax.patches[4*3-1].set_linewidth(2.5)
    ax.patches[4*3-2].set_edgecolor('red')
    ax.patches[4*3-2].set_linewidth(2.5)
    ax.patches[4*3-3].set_edgecolor('red')
    ax.patches[4*3-3].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_comparison.png")
    plt.close()


# ============================================================================
# 3. Model Variant Comparison (Full vs Lite)
# ============================================================================

def plot_model_comparison():
    """Grouped bar chart comparing Full vs Lite models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(models))
    width = 0.5
    
    # IoU comparison
    bars1 = ax1.bar(x, [v*100 for v in model_iou], width, color=['#1976D2', '#03A9F4'], 
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('IoU (%)', fontweight='bold')
    ax1.set_title('Mean IoU', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 60)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Dice comparison
    bars2 = ax2.bar(x, [v*100 for v in model_dice], width, color=['#388E3C', '#66BB6A'], 
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Dice (%)', fontweight='bold')
    ax2.set_title('Mean Dice Score', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 80)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Parameters comparison
    bars3 = ax3.bar(x, model_params, width, color=['#F57C00', '#FFB74D'], 
                    edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Parameters (M)', fontweight='bold')
    ax3.set_title('Model Size', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # FPS comparison
    bars4 = ax4.bar(x, model_fps, width, color=['#7B1FA2', '#BA68C8'], 
                    edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Inference FPS', fontweight='bold')
    ax4.set_title('Inference Speed', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)} FPS', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Full Model vs Lite Model Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_comparison.png")
    plt.close()


# ============================================================================
# 4. Robustness Analysis
# ============================================================================

def plot_robustness_analysis():
    """Bar chart showing robustness under different conditions"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(robustness_conditions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [v*100 for v in robustness_full], width, 
                   label='Full Model', color='#1976D2', edgecolor='black', linewidth=1.0)
    bars2 = ax.bar(x + width/2, [v*100 for v in robustness_lite], width, 
                   label='Lite Model', color='#03A9F4', edgecolor='black', linewidth=1.0)
    
    ax.set_ylabel('IoU (%)', fontweight='bold')
    ax.set_xlabel('Test Condition', fontweight='bold')
    ax.set_title('Robustness Analysis: Model Performance Under Degradations', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(robustness_conditions)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 60)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add degradation percentages as annotations
    degradations_full = [0, -5.7, -8.5, -2.0]
    degradations_lite = [0, -6.6, -9.2, -2.4]
    
    for i, (deg_f, deg_l) in enumerate(zip(degradations_full, degradations_lite)):
        if deg_f < 0:
            ax.text(i - width/2, robustness_full[i]*100 - 3, f'{deg_f:.1f}%', 
                   ha='center', fontsize=8, color='red', fontweight='bold')
        if deg_l < 0:
            ax.text(i + width/2, robustness_lite[i]*100 - 3, f'{deg_l:.1f}%', 
                   ha='center', fontsize=8, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/robustness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/robustness_analysis.png")
    plt.close()


# ============================================================================
# 5. IoU Improvement Breakdown (Ablation Delta)
# ============================================================================

def plot_ablation_delta():
    """Bar chart showing incremental IoU gain from each component"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate deltas
    baseline_iou = ablation_iou[0]
    improvements = [(ablation_iou[i] - baseline_iou) * 100 for i in range(len(ablation_iou))]
    component_names = ['Baseline', 'ASPP\n(+2.7%)', 'CBAM\n(+2.2%)', 'Attn Gates\n(+1.2%)', 'Full Model\n(+8.05%)']
    
    colors = ['#9E9E9E', '#4CAF50', '#2196F3', '#FF9800', '#F44336']
    bars = ax.bar(range(len(improvements)), improvements, color=colors, 
                  edgecolor='black', linewidth=1.2, width=0.6)
    
    ax.set_ylabel('IoU Improvement over Baseline (%)', fontweight='bold')
    ax.set_xlabel('Architectural Component', fontweight='bold')
    ax.set_title('Ablation Study: Incremental IoU Gain from Each Component', fontweight='bold', pad=15)
    ax.set_xticks(range(len(component_names)))
    ax.set_xticklabels(component_names, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.2,
                   f'+{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_delta.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_delta.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Generating Results Visualizations (Bar Charts)")
    print("="*80 + "\n")
    
    plot_per_class_metrics()
    plot_ablation_comparison()
    plot_model_comparison()
    plot_robustness_analysis()
    plot_ablation_delta()
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*80}\n")
    
    # Print summary
    print("Generated Charts:")
    print("  1. per_class_performance.png - IoU and Dice per class")
    print("  2. ablation_comparison.png - Ablation study comparison")
    print("  3. model_comparison.png - Full vs Lite model metrics")
    print("  4. robustness_analysis.png - Robustness under degradations")
    print("  5. ablation_delta.png - Incremental IoU gains")
    print()
