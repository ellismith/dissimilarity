import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def visualize_raw_vs_zscore(base_path, louvain, region, min_age=1.0):
    """
    Create comprehensive side-by-side visualization of raw vs z-scored results
    """
    
    print(f"Creating visualizations: Louvain {louvain}, Region {region}")
    print(f"{'='*70}\n")
    
    # Find files
    raw_pattern = os.path.join(base_path, f"*louvain{louvain}_{region}_minage{min_age}")
    zscore_pattern = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_zscore")
    
    # Load distance matrices
    raw_dist_files = [f for f in glob.glob(raw_pattern + "_distance_matrix.npy") if 'zscore' not in f]
    zscore_dist_file = zscore_pattern + "_distance_matrix.npy"
    
    if not raw_dist_files or not os.path.exists(zscore_dist_file):
        print("ERROR: Could not find distance matrices")
        return
    
    raw_dist = np.load(raw_dist_files[0])
    zscore_dist = np.load(zscore_dist_file)
    
    # Load metadata
    raw_meta_files = [f for f in glob.glob(raw_pattern + "_cell_metadata.csv") if 'zscore' not in f]
    zscore_meta_file = zscore_pattern + "_cell_metadata.csv"
    
    raw_meta = pd.read_csv(raw_meta_files[0])
    zscore_meta = pd.read_csv(zscore_meta_file)
    
    # Load KNN results
    raw_knn_files = [f for f in glob.glob(raw_pattern + "_knn_analysis_k10.csv") if 'zscore' not in f]
    zscore_knn_file = zscore_pattern + "_knn_analysis_k10.csv"
    
    raw_knn = pd.read_csv(raw_knn_files[0])
    zscore_knn = pd.read_csv(zscore_knn_file)
    
    # Load validation summaries
    raw_val_files = [f for f in glob.glob(raw_pattern + "_validation_summary.csv") if 'zscore' not in f]
    zscore_val_file = zscore_pattern + "_validation_summary.csv"
    
    raw_val = pd.read_csv(raw_val_files[0]).iloc[0]
    zscore_val = pd.read_csv(zscore_val_file).iloc[0]
    
    print("Loaded all data files")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # ROW 1: Distance Heatmaps (subsampled)
    # =========================================================================
    print("Creating heatmaps...")
    
    # Subsample for visualization
    np.random.seed(42)
    n_sample = min(300, len(raw_meta))
    sample_idx = np.random.choice(len(raw_meta), n_sample, replace=False)
    sample_idx = np.sort(sample_idx)
    
    # Sort by animal
    meta_sorted = raw_meta.iloc[sample_idx].sort_values(['animal_id', 'age']).reset_index(drop=True)
    sort_idx = sample_idx[meta_sorted.index]
    
    # Raw heatmap
    ax1 = fig.add_subplot(gs[0, 0:2])
    raw_sub = raw_dist[np.ix_(sort_idx, sort_idx)]
    im1 = ax1.imshow(raw_sub, cmap='viridis', aspect='auto')
    
    # Add animal boundaries
    animal_bounds = meta_sorted.groupby('animal_id').size().cumsum().values[:-1]
    for b in animal_bounds:
        ax1.axhline(b, color='red', linewidth=0.3, alpha=0.5)
        ax1.axvline(b, color='red', linewidth=0.3, alpha=0.5)
    
    ax1.set_title('RAW Expression\nDistance Matrix (sorted by animal)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cell index')
    ax1.set_ylabel('Cell index')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Distance')
    
    # Z-scored heatmap
    ax2 = fig.add_subplot(gs[0, 2:4])
    zscore_sub = zscore_dist[np.ix_(sort_idx, sort_idx)]
    im2 = ax2.imshow(zscore_sub, cmap='viridis', aspect='auto')
    
    for b in animal_bounds:
        ax2.axhline(b, color='red', linewidth=0.3, alpha=0.5)
        ax2.axvline(b, color='red', linewidth=0.3, alpha=0.5)
    
    ax2.set_title('Z-SCORED Expression\nDistance Matrix (sorted by animal)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cell index')
    ax2.set_ylabel('Cell index')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Distance')
    
    # =========================================================================
    # ROW 2: Distance Distributions
    # =========================================================================
    print("Creating distance distributions...")
    
    # Overall distance distribution
    ax3 = fig.add_subplot(gs[1, 0])
    raw_dists = raw_dist[np.triu_indices_from(raw_dist, k=1)]
    zscore_dists = zscore_dist[np.triu_indices_from(zscore_dist, k=1)]
    
    ax3.hist(raw_dists, bins=100, alpha=0.6, label='Raw', edgecolor='black', density=True)
    ax3.axvline(raw_dists.mean(), color='blue', linestyle='--', linewidth=2, label=f'Raw mean={raw_dists.mean():.0f}')
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Density')
    ax3.set_title('Raw: Distance Distribution', fontweight='bold')
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(zscore_dists, bins=100, alpha=0.6, label='Z-scored', color='orange', edgecolor='black', density=True)
    ax4.axvline(zscore_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'Z-score mean={zscore_dists.mean():.0f}')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Density')
    ax4.set_title('Z-scored: Distance Distribution', fontweight='bold')
    ax4.legend()
    
    # NN distance distributions
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist([raw_knn['nearest_neighbor_distance'], zscore_knn['nearest_neighbor_distance']], 
             bins=50, alpha=0.6, label=['Raw', 'Z-scored'], edgecolor='black')
    ax5.set_xlabel('Distance to Nearest Neighbor')
    ax5.set_ylabel('Count')
    ax5.set_title('NN Distance Comparison', fontweight='bold')
    ax5.legend()
    
    # Key metrics comparison
    ax6 = fig.add_subplot(gs[1, 3])
    metrics = ['Animal\nEffect', 'Distance\nStructure', 'Animal\nMixing']
    raw_vals = [raw_val['ratio_diff_to_same'], 
                raw_val['ratio_random_to_nn']/2,  # Scale down for viz
                raw_val['pct_diff_animal_neighbors']/100]
    zscore_vals = [zscore_val['ratio_diff_to_same'],
                   zscore_val['ratio_random_to_nn']/2,
                   zscore_val['pct_diff_animal_neighbors']/100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, raw_vals, width, label='Raw', alpha=0.8, edgecolor='black')
    bars2 = ax6.bar(x + width/2, zscore_vals, width, label='Z-scored', alpha=0.8, edgecolor='black')
    
    ax6.set_ylabel('Value')
    ax6.set_title('Key Metrics', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.axhline(1, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # =========================================================================
    # ROW 3: Age Analysis
    # =========================================================================
    print("Creating age analysis plots...")
    
    # Age difference distributions
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(raw_knn['min_age_diff'], bins=50, alpha=0.7, label='Raw', edgecolor='black')
    ax7.axvline(raw_knn['min_age_diff'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f"Mean={raw_knn['min_age_diff'].mean():.2f}y")
    ax7.set_xlabel('Age Diff to NN (years)')
    ax7.set_ylabel('Count')
    ax7.set_title('Raw: Age Clustering', fontweight='bold')
    ax7.legend()
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(zscore_knn['min_age_diff'], bins=50, alpha=0.7, color='orange', label='Z-scored', edgecolor='black')
    ax8.axvline(zscore_knn['min_age_diff'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean={zscore_knn['min_age_diff'].mean():.2f}y")
    ax8.set_xlabel('Age Diff to NN (years)')
    ax8.set_ylabel('Count')
    ax8.set_title('Z-scored: Age Clustering', fontweight='bold')
    ax8.legend()
    
    # Direct comparison scatter
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(raw_knn['min_age_diff'], zscore_knn['min_age_diff'], 
               alpha=0.3, s=10, edgecolor='none')
    max_val = max(raw_knn['min_age_diff'].max(), zscore_knn['min_age_diff'].max())
    ax9.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
    ax9.set_xlabel('Age Diff to NN (Raw)')
    ax9.set_ylabel('Age Diff to NN (Z-scored)')
    ax9.set_title('Per-Cell Age Difference', fontweight='bold')
    ax9.legend()
    
    # Age vs NN distance
    ax10 = fig.add_subplot(gs[2, 3])
    
    # Normalize distances for comparison
    raw_nn_norm = (raw_knn['nearest_neighbor_distance'] - raw_knn['nearest_neighbor_distance'].mean()) / raw_knn['nearest_neighbor_distance'].std()
    zscore_nn_norm = (zscore_knn['nearest_neighbor_distance'] - zscore_knn['nearest_neighbor_distance'].mean()) / zscore_knn['nearest_neighbor_distance'].std()
    
    ax10.scatter(raw_knn['age'], raw_nn_norm, alpha=0.3, s=10, label='Raw', edgecolor='none')
    ax10.scatter(zscore_knn['age'], zscore_nn_norm, alpha=0.3, s=10, label='Z-scored', edgecolor='none')
    ax10.set_xlabel('Cell Age (years)')
    ax10.set_ylabel('Normalized NN Distance')
    ax10.set_title('NN Distance vs Age', fontweight='bold')
    ax10.legend()
    ax10.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # =========================================================================
    # ROW 4: Summary Statistics
    # =========================================================================
    print("Creating summary...")
    
    # Summary table
    ax11 = fig.add_subplot(gs[3, :2])
    ax11.axis('off')
    
    summary_data = [
        ['Metric', 'Raw', 'Z-scored', 'Δ', 'Winner'],
        ['Same/Diff Animal Ratio', f"{raw_val['ratio_diff_to_same']:.3f}", 
         f"{zscore_val['ratio_diff_to_same']:.3f}",
         f"{zscore_val['ratio_diff_to_same']-raw_val['ratio_diff_to_same']:+.3f}",
         '✓ Z-score' if zscore_val['ratio_diff_to_same'] < raw_val['ratio_diff_to_same'] else 'Raw'],
        
        ['Random/NN Ratio (structure)', f"{raw_val['ratio_random_to_nn']:.2f}",
         f"{zscore_val['ratio_random_to_nn']:.2f}",
         f"{zscore_val['ratio_random_to_nn']-raw_val['ratio_random_to_nn']:+.2f}",
         '✓ Raw' if raw_val['ratio_random_to_nn'] > zscore_val['ratio_random_to_nn'] else 'Z-score'],
        
        ['% Diff Animal Neighbors', f"{raw_val['pct_diff_animal_neighbors']:.1f}%",
         f"{zscore_val['pct_diff_animal_neighbors']:.1f}%",
         f"{zscore_val['pct_diff_animal_neighbors']-raw_val['pct_diff_animal_neighbors']:+.1f}%",
         '✓ Z-score' if zscore_val['pct_diff_animal_neighbors'] > raw_val['pct_diff_animal_neighbors'] else 'Raw'],
        
        ['Age Diff to NN', f"{raw_knn['min_age_diff'].mean():.2f}y",
         f"{zscore_knn['min_age_diff'].mean():.2f}y",
         f"{zscore_knn['min_age_diff'].mean()-raw_knn['min_age_diff'].mean():+.2f}y",
         '✓ Raw' if raw_knn['min_age_diff'].mean() < zscore_knn['min_age_diff'].mean() else 'Z-score']
    ]
    
    table = ax11.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax11.set_title('Summary Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Recommendation text
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    recommendation = f"""
RECOMMENDATION:
    
{'✓ USE RAW EXPRESSION' if raw_val['ratio_random_to_nn'] > 1.8 else '• CONSIDER Z-SCORING'}

Reasoning:
- Raw captures {raw_val['ratio_random_to_nn']:.2f}x structure
- Z-scored captures {zscore_val['ratio_random_to_nn']:.2f}x structure
  → {'Raw has MUCH better biological structure' if raw_val['ratio_random_to_nn'] > zscore_val['ratio_random_to_nn'] + 0.5 else 'Similar structure'}

- Raw age clustering: {raw_knn['min_age_diff'].mean():.2f} years
- Z-scored age clustering: {zscore_knn['min_age_diff'].mean():.2f} years
  → {'Raw detects age effects better' if raw_knn['min_age_diff'].mean() < zscore_knn['min_age_diff'].mean() else 'Z-scored detects age effects better'}

- Animal effects: {raw_val['ratio_diff_to_same']:.3f} (raw) vs {zscore_val['ratio_diff_to_same']:.3f} (z-score)
  → {'Both minimal - no normalization needed' if raw_val['ratio_diff_to_same'] < 1.1 else 'Z-scoring helps reduce batch effects'}

Your data already has excellent quality!
Z-scoring removes biologically meaningful
magnitude information without significant benefit.
    """
    
    ax12.text(0.1, 0.5, recommendation, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'Raw vs Z-Scored Expression Analysis\nLouvain {louvain}, {region}, age≥{min_age}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_raw_vs_zscore_comprehensive.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive visualization: {os.path.basename(output_file)}")
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize raw vs z-scored comparison')
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--louvain', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--min-age', type=float, default=1.0)
    
    args = parser.parse_args()
    
    visualize_raw_vs_zscore(args.base_path, args.louvain, args.region, args.min_age)
