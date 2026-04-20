#!/usr/bin/env python
"""
Plot the actual pseudobulk PC scores per animal for a subtype
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='Plot pseudobulk PC scores for a subtype')
    parser.add_argument('--cell_type', type=str, required=True,
                        help='Cell type (e.g., Glutamatergic)')
    parser.add_argument('--region', type=str, required=True,
                        help='Brain region (e.g., CN)')
    parser.add_argument('--subtype', type=str, required=True,
                        help='Subtype (e.g., glutamatergic neurons_21)')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--pc_scores_path', type=str, required=True,
                        help='Path to PC scores CSV')
    parser.add_argument('--subtype_col', type=str, default='ct_louvain',
                        help='Column name for subtypes')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--min_age', type=float, default=1.0,
                        help='Minimum age to include')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print(f"PLOTTING PSEUDOBULK PC SCORES")
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
    print(f"Found {len(pc_cols)} PCs")
    
    # Figure 1: PC1 vs PC2 scatter
    print("\nCreating Figure 1: PC1 vs PC2")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(subtype_data['PC1'], subtype_data['PC2'], 
                        c=subtype_data['age'], cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add animal labels
    for _, row in subtype_data.iterrows():
        ax.annotate(f"{row['age']:.0f}", 
                   (row['PC1'], row['PC2']),
                   fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title(f'{args.subtype} in {args.region}\nPseudobulk PC1 vs PC2 (N={len(subtype_data)} animals)', 
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age (years)', fontsize=12)
    
    # Add centroid
    centroid = (subtype_data['PC1'].mean(), subtype_data['PC2'].mean())
    ax.scatter(*centroid, s=400, c='red', marker='*', 
              edgecolors='black', linewidth=2, label='Centroid', zorder=5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype.replace(' ', '_')}_pseudobulk_PC1_PC2.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved PC1 vs PC2 plot")
    plt.close()
    
    # Figure 2: Each PC vs age (top 6 PCs)
    print("\nCreating Figure 2: PC scores vs age")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, pc in enumerate(pc_cols[:6]):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(subtype_data['age'], subtype_data[pc], 
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Fit line
        z = np.polyfit(subtype_data['age'], subtype_data[pc], 1)
        p = np.poly1d(z)
        ax.plot(subtype_data['age'], p(subtype_data['age']), 
               "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr, pval = stats.spearmanr(subtype_data['age'], subtype_data[pc])
        
        ax.set_xlabel('Age (years)', fontsize=10)
        ax.set_ylabel(f'{pc} score', fontsize=10)
        ax.set_title(f'{pc}\nr={corr:.3f}, p={pval:.3f}', fontsize=11, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    fig.suptitle(f'{args.subtype} in {args.region}\nPC Scores vs Age', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype.replace(' ', '_')}_pseudobulk_PCs_vs_age.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved PC vs age plots")
    plt.close()
    
    # Figure 3: Absolute deviation from mean vs age
    print("\nCreating Figure 3: Variability (deviation from mean)")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, pc in enumerate(pc_cols[:6]):
        ax = axes[i]
        
        # Calculate absolute deviation from mean
        pc_mean = subtype_data[pc].mean()
        abs_dev = np.abs(subtype_data[pc] - pc_mean)
        
        # Scatter plot
        ax.scatter(subtype_data['age'], abs_dev, 
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Fit line
        z = np.polyfit(subtype_data['age'], abs_dev, 1)
        p = np.poly1d(z)
        ax.plot(subtype_data['age'], p(subtype_data['age']), 
               "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr, pval = stats.spearmanr(subtype_data['age'], abs_dev)
        
        ax.set_xlabel('Age (years)', fontsize=10)
        ax.set_ylabel(f'|{pc} - mean|', fontsize=10)
        ax.set_title(f'{pc} Variability\nr={corr:.3f}, p={pval:.3f}', 
                    fontsize=11, fontweight='bold')
    
    fig.suptitle(f'{args.subtype} in {args.region}\nAbsolute Deviation from Mean (Variability Measure)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype.replace(' ', '_')}_pseudobulk_variability.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved variability plots")
    plt.close()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
