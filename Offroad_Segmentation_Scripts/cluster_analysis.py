"""
Cluster Analysis and Visualization for Segmentation Results
Analyzes segmented video to create:
  1. Class distribution clusters over time
  2. Spatial clustering of terrain regions
  3. Color-coded cluster visualization
  4. Temporal clustering patterns
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from collections import defaultdict
import argparse

# Class definitions
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

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


def extract_class_from_color(frame):
    """Extract class labels from color-coded segmentation frame"""
    h, w = frame.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    for class_id, color in enumerate(color_palette):
        # Find pixels matching this color
        mask = np.all(frame == color, axis=-1)
        class_mask[mask] = class_id
    
    return class_mask


def analyze_video_clustering(video_path, output_dir='cluster_analysis'):
    """Analyze segmentation video and create cluster visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Analyzing {total_frames} frames from {video_path}...")
    
    # Data structures for analysis
    frame_distributions = []  # Class distribution per frame
    spatial_features = []     # Spatial clustering features
    frame_indices = []
    
    frame_idx = 0
    sample_rate = max(1, total_frames // 100)  # Sample every N frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Extract class mask
            class_mask = extract_class_from_color(frame)
            
            # Calculate class distribution (percentage of each class)
            unique, counts = np.unique(class_mask, return_counts=True)
            total_pixels = class_mask.size
            distribution = np.zeros(len(class_names))
            
            for cls_id, count in zip(unique, counts):
                distribution[cls_id] = count / total_pixels * 100
            
            frame_distributions.append(distribution)
            frame_indices.append(frame_idx)
            
            # Extract spatial features (mean position of each class)
            spatial_feat = []
            for cls_id in range(len(class_names)):
                mask = class_mask == cls_id
                if mask.any():
                    y_coords, x_coords = np.where(mask)
                    mean_y = np.mean(y_coords) / class_mask.shape[0]  # Normalize
                    mean_x = np.mean(x_coords) / class_mask.shape[1]
                    spatial_feat.extend([mean_y, mean_x])
                else:
                    spatial_feat.extend([0, 0])
            
            spatial_features.append(spatial_feat)
        
        frame_idx += 1
    
    cap.release()
    
    frame_distributions = np.array(frame_distributions)
    spatial_features = np.array(spatial_features)
    
    print(f"Extracted {len(frame_distributions)} sampled frames")
    
    # ========================================================================
    # 1. Temporal Class Distribution Heatmap
    # ========================================================================
    plt.figure(figsize=(16, 8))
    plt.imshow(frame_distributions.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Percentage (%)')
    plt.xlabel('Frame Index', fontweight='bold')
    plt.ylabel('Terrain Class', fontweight='bold')
    plt.title('Temporal Class Distribution Heatmap', fontweight='bold', fontsize=14)
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/temporal_heatmap.png")
    plt.close()
    
    # ========================================================================
    # 2. Cluster Analysis of Frame Patterns (K-Means on distributions)
    # ========================================================================
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(frame_distributions)
    
    # Plot clusters in PCA space
    pca = PCA(n_components=2)
    frame_pca = pca.fit_transform(frame_distributions)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(frame_pca[:, 0], frame_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', s=50, alpha=0.7, edgecolor='black')
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
    plt.title('Frame Clusters (K-Means on Class Distributions)', fontweight='bold', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add cluster centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='X', s=300, edgecolor='black', linewidth=2, label='Centroids')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_pca.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cluster_pca.png")
    plt.close()
    
    # ========================================================================
    # 3. Cluster Profiles (Average class distribution per cluster)
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_frames = frame_distributions[cluster_mask]
        mean_distribution = cluster_frames.mean(axis=0)
        
        ax = axes[cluster_id]
        colors = ['#' + ''.join([f'{c:02x}' for c in color]) for color in color_palette]
        bars = ax.barh(class_names, mean_distribution, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_xlabel('Average Percentage (%)', fontweight='bold')
        ax.set_title(f'Cluster {cluster_id} ({cluster_mask.sum()} frames)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(50, mean_distribution.max() + 5))
        
        # Add value labels
        for bar, val in zip(bars, mean_distribution):
            if val > 1:
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Cluster Profiles: Average Class Distribution per Cluster', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_profiles.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cluster_profiles.png")
    plt.close()
    
    # ========================================================================
    # 4. Cluster Timeline (Show which cluster each frame belongs to)
    # ========================================================================
    plt.figure(figsize=(16, 4))
    plt.scatter(frame_indices, cluster_labels, c=cluster_labels, cmap='tab10', 
               s=20, alpha=0.6, edgecolor='none')
    plt.xlabel('Frame Index', fontweight='bold')
    plt.ylabel('Cluster ID', fontweight='bold')
    plt.title('Cluster Timeline (Scene Type Over Time)', fontweight='bold', fontsize=14)
    plt.yticks(range(n_clusters))
    plt.grid(alpha=0.3, linestyle='--')
    plt.colorbar(label='Cluster ID')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_timeline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cluster_timeline.png")
    plt.close()
    
    # ========================================================================
    # 5. Overall Class Distribution Summary
    # ========================================================================
    overall_distribution = frame_distributions.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = ['#' + ''.join([f'{c:02x}' for c in color]) for color in color_palette]
    bars = ax1.barh(class_names, overall_distribution, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Average Percentage (%)', fontweight='bold')
    ax1.set_title('Overall Average Class Distribution', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, overall_distribution):
        if val > 0.5:
            ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Pie chart
    # Only show classes with >1% presence
    significant_indices = overall_distribution > 1.0
    sig_names = [name for i, name in enumerate(class_names) if significant_indices[i]]
    sig_values = overall_distribution[significant_indices]
    sig_colors = [colors[i] for i in range(len(colors)) if significant_indices[i]]
    
    ax2.pie(sig_values, labels=sig_names, colors=sig_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Class Distribution (>1% classes)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/overall_distribution.png")
    plt.close()
    
    # ========================================================================
    # 6. Generate Summary Report
    # ========================================================================
    with open(f'{output_dir}/cluster_analysis_report.txt', 'w') as f:
        f.write("CLUSTER ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total Frames Analyzed: {len(frame_distributions)}\n")
        f.write(f"Number of Clusters: {n_clusters}\n\n")
        
        f.write("Overall Class Distribution:\n")
        f.write("-" * 80 + "\n")
        for name, pct in zip(class_names, overall_distribution):
            f.write(f"  {name:<20}: {pct:>6.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Cluster Analysis:\n")
        f.write("-" * 80 + "\n")
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_frames = frame_distributions[cluster_mask]
            mean_dist = cluster_frames.mean(axis=0)
            
            f.write(f"\nCluster {cluster_id} ({cluster_mask.sum()} frames):\n")
            dominant_classes = np.argsort(mean_dist)[::-1][:3]
            f.write(f"  Dominant Classes: {', '.join([class_names[i] for i in dominant_classes])}\n")
            f.write(f"  Top class percentages: " + 
                   ", ".join([f"{class_names[i]}: {mean_dist[i]:.1f}%" for i in dominant_classes]) + "\n")
    
    print(f"✓ Saved: {output_dir}/cluster_analysis_report.txt")
    
    print(f"\n{'='*80}")
    print("Cluster Analysis Complete!")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster analysis of segmentation video')
    parser.add_argument('--video', default='../vecteezy_output_segmented.mp4', 
                       help='Path to segmented video')
    parser.add_argument('--output_dir', default='cluster_analysis',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    analyze_video_clustering(args.video, args.output_dir)
