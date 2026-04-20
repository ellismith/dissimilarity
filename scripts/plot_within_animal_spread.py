#!/usr/bin/env python
"""
Measure how spread out cells from the same animal are in PC space
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='Measure within-animal cell spread')
    parser.add_argument('--cell_type', type=str, required=True,
                        help='Cell type (e.g., Microglia)')
    parser.add_argument('--region', type=str, required=True,
                        help='Brain region (e.g., EC)')
    parser.add_argument('--subtype', type=str, required=True,
                        help='Subtype (e.g., microglia_9)')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--subtype_col', type=str, default='ct_louvain',
                        help='Column name for subtypes')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--min_age', type=float, default=1.0,
                        help='Minimum age to include')
    parser.add_argument('--min_cells', type=int, default=10,
                        help='Minimum cells per animal')
    parser.add_argument('--n_pcs', type=int, default=20,
                        help='Number of PCs to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print(f"MEASURING WITHIN-ANIMAL CELL SPREAD")
    print(f"Cell type: {args.cell_type}")
    print(f"Region: {args.region}")
    print(f"Subtype: {args.subtype}")
    print("="*70)
    
    # Load h5ad
    print(f"\nLoading h5ad...")
    adata = sc.read_h5ad(args.h5ad_path, backed='r')
    
    # Filter to region, subtype, age
    print("Filtering cells...")
    obs_df = adata.obs.copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    
    mask = (obs_df['region'] == args.region) & \
           (obs_df[args.subtype_col] == args.subtype) & \
           (obs_df['age'] >= args.min_age)
    
    cell_indices = np.where(mask)[0]
    print(f"Found {len(cell_indices)} cells matching criteria")
    
    if len(cell_indices) == 0:
        print("ERROR: No cells found!")
        return
    
    # Get metadata for these cells
    metadata = obs_df.iloc[cell_indices][['animal_id', 'age', 'sex']].copy()
    
    # Check if PCA already exists
    if 'X_pca' not in adata.obsm.keys():
        print("\nERROR: No PCA found in h5ad! Need to run PCA first.")
        print("The h5ad file needs to have PCA computed and stored in .obsm['X_pca']")
        return
    
    # Get PCA coordinates for these cells
    print("Extracting PC coordinates...")
    pca_coords = adata.obsm['X_pca'][cell_indices, :args.n_pcs]
    print(f"Using {args.n_pcs} PCs")
    
    # Calculate within-animal spread
    print("\nCalculating within-animal spread...")
    animal_stats = []
    
    for animal_id in metadata['animal_id'].unique():
        animal_mask = metadata['animal_id'] == animal_id
        n_cells = animal_mask.sum()
        
        if n_cells < args.min_cells:
            continue
        
        # Get this animal's cells
        animal_coords = pca_coords[animal_mask]
        
        # Calculate centroid
        centroid = animal_coords.mean(axis=0)
        
        # Calculate distances from centroid
        distances = np.sqrt(((animal_coords - centroid) ** 2).sum(axis=1))
        
        # Calculate stats
        animal_stats.append({
            'animal_id': animal_id,
            'age': metadata[animal_mask]['age'].iloc[0],
            'sex': metadata[animal_mask]['sex'].iloc[0],
            'n_cells': n_cells,
            'mean_distance': distances.mean(),
            'median_distance': np.median(distances),
            'std_distance': distances.std(),
            'max_distance': distances.max()
        })
    
    stats_df = pd.DataFrame(animal_stats)
    print(f"\nCalculated spread for {len(stats_df)} animals")
    
    if len(stats_df) < 3:
        print("ERROR: Not enough animals to analyze!")
        return
    
    # Figure 1: Mean within-animal spread vs age
    print("\nCreating Figure 1: Within-animal spread vs age")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = ['mean_distance', 'median_distance', 'std_distance', 'max_distance']
    titles = ['Mean Distance from Centroid', 'Median Distance from Centroid', 
              'Std Dev of Distances', 'Max Distance from Centroid']
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        # Scatter plot
        ax.scatter(stats_df['age'], stats_df[metric], s=100, alpha=0.7, 
                  c=stats_df['n_cells'], cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Fit line
        z = np.polyfit(stats_df['age'], stats_df[metric], 1)
        p = np.poly1d(z)
        ax.plot(stats_df['age'], p(stats_df['age']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr, pval = stats.spearmanr(stats_df['age'], stats_df[metric])
        
        ax.set_xlabel('Age (years)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\nr={corr:.3f}, p={pval:.3e}', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'{args.subtype} in {args.region}\nWithin-Animal Cell Spread in PC Space (PC1-{args.n_pcs})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_within_animal_spread.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_within_animal_spread.png")
    plt.close()
    
    # Figure 2: Example animals (young vs old)
    print("\nCreating Figure 2: Example young vs old animals")
    
    # Get youngest and oldest animals
    young_animal = stats_df.nsmallest(1, 'age')['animal_id'].iloc[0]
    old_animal = stats_df.nlargest(1, 'age')['animal_id'].iloc[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, animal_id, label in zip(axes, [young_animal, old_animal], ['Youngest', 'Oldest']):
        animal_mask = metadata['animal_id'] == animal_id
        animal_coords = pca_coords[animal_mask]
        animal_age = metadata[animal_mask]['age'].iloc[0]
        
        # Plot PC1 vs PC2
        ax.scatter(animal_coords[:, 0], animal_coords[:, 1], s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot centroid
        centroid = animal_coords.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], s=300, c='red', marker='*', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'{label} Animal (Age={animal_age:.1f})\nN={animal_mask.sum()} cells',
                    fontsize=12, fontweight='bold')
    
    fig.suptitle(f'{args.subtype} in {args.region}\nCell Distribution in PC Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_young_vs_old_cells.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_young_vs_old_cells.png")
    plt.close()
    
    # Save stats
    stats_df.to_csv(f"{args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_within_animal_stats.csv", 
                    index=False)
    print(f"\nSaved stats: {args.output_dir}/{args.cell_type}_{args.region}_{args.subtype}_within_animal_stats.csv")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
