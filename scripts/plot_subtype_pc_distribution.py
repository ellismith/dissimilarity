#!/usr/bin/env python
"""
Plot the actual distribution of PC scores for a specific subtype in a region
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Plot PC score distribution for a subtype')
    parser.add_argument('--cell_type', type=str, required=True,
                        help='Cell type (e.g., Microglia)')
    parser.add_argument('--region', type=str, required=True,
                        help='Brain region (e.g., EC)')
    parser.add_argument('--subtype', type=str, required=True,
                        help='Subtype (e.g., microglia_9)')
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
    print(f"PLOTTING PC DISTRIBUTION")
    print(f"Cell type: {args.cell_type}")
    print(f"Region: {args.region}")
    print(f"Subtype: {args.subtype}")
    print("="*70)
    
    # Load h5ad to get subtype annotations
    print(f"\nLoading h5ad...")
    adata = sc.read_h5ad(args.h5ad_path, backed='r')
    
    # Get animal-subtype mapping
    obs_df = adata.obs[['animal_id', 'region', 'age', args.subtype_col]].copy()
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
    
    # Create age bins for visualization
    subtype_data['age_group'] = pd.cut(subtype_data['age'], 
                                        bins=[0, 7, 14, 100],
                                        labels=['Young (1-7)', 'Middle (7-14)', 'Old (14+)'])
    
    # Figure 1: PC1 vs PC2 colored by age
    print("\nCreating Figure 1: PC1 vs PC2")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(subtype_data['PC1'], subtype_data['PC2'], 
                        c=subtype_data['age'], cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{args.subtype} in {args.region}\nPC1 vs PC2 (colored by age)', 
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age (years)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_PC1_PC2.png", 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_PC1_PC2.png")
    plt.close()
    
    # Figure 2: Distribution of each PC by age group
    print("\nCreating Figure 2: PC distributions by age group")
    n_pcs = len(pc_cols)
    n_rows = (n_pcs + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, pc in enumerate(pc_cols):
        for age_group in ['Young (1-7)', 'Middle (7-14)', 'Old (14+)']:
            data = subtype_data[subtype_data['age_group'] == age_group][pc]
            if len(data) > 0:
                axes[i].hist(data, alpha=0.5, label=age_group, bins=10)
        
        axes[i].set_xlabel(pc, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].legend()
        axes[i].set_title(f'{pc} distribution')
    
    # Remove extra subplots
    for i in range(n_pcs, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f'{args.subtype} in {args.region}\nPC Distributions by Age Group', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_PC_distributions.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_PC_distributions.png")
    plt.close()
    
    # Figure 3: Spread (distance from centroid) vs age
    print("\nCreating Figure 3: Spread vs age")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate distance from centroid in PC space (using first 5 PCs)
    pc_matrix = subtype_data[pc_cols[:5]].values
    centroid = pc_matrix.mean(axis=0)
    distances = np.sqrt(((pc_matrix - centroid) ** 2).sum(axis=1))
    
    ax.scatter(subtype_data['age'], distances, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Fit line
    z = np.polyfit(subtype_data['age'], distances, 1)
    p = np.poly1d(z)
    ax.plot(subtype_data['age'], p(subtype_data['age']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    from scipy import stats
    corr, pval = stats.spearmanr(subtype_data['age'], distances)
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Distance from centroid (PC1-5)', fontsize=12)
    ax.set_title(f'{args.subtype} in {args.region}\nSpread in PC space vs Age\nr={corr:.3f}, p={pval:.3e}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_spread_vs_age.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_spread_vs_age.png")
    plt.close()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
