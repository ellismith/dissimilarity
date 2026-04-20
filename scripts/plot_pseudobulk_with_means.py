#!/usr/bin/env python
"""
Plot pseudobulk PC scores with means highlighted
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='Plot pseudobulk PC scores with means')
    parser.add_argument('--cell_type', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--subtype', type=str, required=True)
    parser.add_argument('--h5ad_path', type=str, required=True)
    parser.add_argument('--pc_scores_path', type=str, required=True)
    parser.add_argument('--subtype_col', type=str, default='ct_louvain')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--min_age', type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print(f"PLOTTING WITH MEANS")
    print(f"Cell type: {args.cell_type}")
    print(f"Region: {args.region}")
    print(f"Subtype: {args.subtype}")
    print("="*70)
    
    # Load h5ad to get subtype annotations
    print(f"\nLoading h5ad...")
    adata = sc.read_h5ad(args.h5ad_path, backed='r')
    
    # Get animal-subtype mapping
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex', args.subtype_col]].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    
    # Filter to region and age
    obs_df = obs_df[(obs_df['region'] == args.region) & (obs_df['age'] >= args.min_age)]
    
    # Get animals with this subtype
    animals_with_subtype = obs_df[obs_df[args.subtype_col] == args.subtype]['animal_id'].unique()
    print(f"Found {len(animals_with_subtype)} animals with {args.subtype} in {args.region}")
    
    # Load PC scores
    print(f"\nLoading PC scores...")
    pc_scores = pd.read_csv(args.pc_scores_path)
    pc_scores['animal_id'] = pc_scores['animal_id'].astype(str)
    pc_scores = pc_scores[pc_scores['age'] >= args.min_age]
    
    # Filter to animals with this subtype
    subtype_data = pc_scores[pc_scores['animal_id'].isin(animals_with_subtype)].copy()
    print(f"Matched {len(subtype_data)} animals")
    
    if len(subtype_data) == 0:
        print("ERROR: No matching animals!")
        return
    
    # Get PC columns
    pc_cols = [col for col in pc_scores.columns if col.startswith('PC')]
    
    # Figure 1: PC1 vs PC2 with mean
    print("\nCreating Figure 1: PC1 vs PC2 with mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(subtype_data['PC1'], subtype_data['PC2'], 
                        c=subtype_data['age'], cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Calculate and plot mean
    mean_pc1 = subtype_data['PC1'].mean()
    mean_pc2 = subtype_data['PC2'].mean()
    ax.scatter(mean_pc1, mean_pc2, s=500, c='red', marker='*', 
              edgecolors='black', linewidth=2, label='Mean', zorder=5)
    
    # Add horizontal and vertical lines at mean
    ax.axhline(mean_pc2, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(mean_pc1, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title(f'{args.subtype} in {args.region}\nPseudobulk PC1 vs PC2 (N={len(subtype_data)} animals)', 
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age (years)', fontsize=12)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype.replace(' ', '_')}_pseudobulk_PC1_PC2_with_mean.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved PC1 vs PC2 plot with mean")
    plt.close()
    
    # Figure 2: Each PC vs age with mean line (top 6 PCs)
    print("\nCreating Figure 2: PC scores vs age with means")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, pc in enumerate(pc_cols[:6]):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(subtype_data['age'], subtype_data[pc], 
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot mean as horizontal line
        mean_pc = subtype_data[pc].mean()
        ax.axhline(mean_pc, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean={mean_pc:.2f}', alpha=0.7)
        
        # Fit line
        z = np.polyfit(subtype_data['age'], subtype_data[pc], 1)
        p = np.poly1d(z)
        ax.plot(subtype_data['age'], p(subtype_data['age']), 
               "blue", linestyle='--', alpha=0.5, linewidth=2, label='Trend')
        
        # Calculate correlation
        corr, pval = stats.spearmanr(subtype_data['age'], subtype_data[pc])
        
        ax.set_xlabel('Age (years)', fontsize=10)
        ax.set_ylabel(f'{pc} score', fontsize=10)
        ax.set_title(f'{pc}\nr={corr:.3f}, p={pval:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    fig.suptitle(f'{args.subtype} in {args.region}\nPC Scores vs Age (with means)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype.replace(' ', '_')}_pseudobulk_PCs_vs_age_with_mean.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved PC vs age plots with means")
    plt.close()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
