#!/usr/bin/env python
"""
Visualize actual deviations from mean for top subtype-PC combinations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats

print("="*70)
print("VISUALIZING ACTUAL DEVIATIONS")
print("="*70)

# Load subtype variability results to find top hits
results_dir = "/scratch/easmit31/factor_analysis/subtype_variability"
import glob

result_files = glob.glob(f"{results_dir}/subtype_variability_*.csv")
all_results = []
for f in result_files:
    df = pd.read_csv(f)
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

# Focus on Glutamatergic and GABAergic
glut_gaba = combined[combined['cell_type'].isin(['Glutamatergic', 'GABAergic'])].copy()

# Get top 12 by absolute correlation
top_hits = glut_gaba.sort_values('abs_dev_corr', key=abs, ascending=False).head(12)

print(f"\nTop 12 subtype-region-PC combinations:")
for idx, row in top_hits.iterrows():
    print(f"  {row['cell_type']} - {row['region']} - {row['subtype']} - {row['PC']}: r={row['abs_dev_corr']:.3f}")

# Paths
glut_path = '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad'
gaba_path = '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'
pc_dir = '/scratch/easmit31/factor_analysis/csv_files'
output_dir = '/scratch/easmit31/factor_analysis/subtype_variability/figures'

# Create 12-panel figure
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for plot_idx, (idx, row) in enumerate(top_hits.iterrows()):
    if plot_idx >= 12:
        break
    
    cell_type = row['cell_type']
    region = row['region']
    subtype = row['subtype']
    pc = row['PC']
    
    print(f"\nProcessing: {cell_type} - {region} - {subtype} - {pc}")
    
    # Load h5ad
    h5ad_path = glut_path if cell_type == 'Glutamatergic' else gaba_path
    
    try:
        adata = sc.read_h5ad(h5ad_path, backed='r')
        
        # Get animals with this subtype
        obs_df = adata.obs[['animal_id', 'region', 'age', 'ct_louvain']].copy()
        obs_df['animal_id'] = obs_df['animal_id'].astype(str)
        obs_df['region'] = obs_df['region'].astype(str)
        
        # Filter
        mask = (obs_df['region'] == region) & (obs_df['age'] >= 1.0) & (obs_df['ct_louvain'] == subtype)
        subtype_obs = obs_df[mask]
        
        animals_with_subtype = subtype_obs['animal_id'].unique()
        
        # Load PC scores
        pc_file = f'{pc_dir}/pca_analysis_{cell_type}_{region}.csv'
        pc_scores = pd.read_csv(pc_file)
        pc_scores['animal_id'] = pc_scores['animal_id'].astype(str)
        pc_scores = pc_scores[pc_scores['age'] >= 1.0]
        
        # Filter to animals with this subtype
        subtype_data = pc_scores[pc_scores['animal_id'].isin(animals_with_subtype)].copy()
        
        if len(subtype_data) < 5:
            print(f"  Skipping - only {len(subtype_data)} animals")
            continue
        
        # Calculate deviations
        pc_mean = subtype_data[pc].mean()
        subtype_data['abs_dev'] = np.abs(subtype_data[pc] - pc_mean)
        
        # Plot
        ax = axes[plot_idx]
        
        # Scatter plot
        scatter = ax.scatter(subtype_data['age'], subtype_data['abs_dev'],
                           c=subtype_data['age'], cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Fit line
        z = np.polyfit(subtype_data['age'], subtype_data['abs_dev'], 1)
        p = np.poly1d(z)
        ax.plot(subtype_data['age'], p(subtype_data['age']), 
               "r--", alpha=0.8, linewidth=2)
        
        # Correlation
        corr = row['abs_dev_corr']
        pval = row['abs_dev_pval']
        
        ax.set_xlabel('Age (years)', fontsize=9)
        ax.set_ylabel(f'|{pc} - mean|', fontsize=9)
        
        # Title with significance
        sig_marker = '***' if row['abs_dev_fdr'] < 0.05 else ''
        ax.set_title(f"{cell_type[:4]}-{region}-{subtype}\n{pc}: r={corr:.3f}, p={pval:.2e} {sig_marker}",
                    fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3)
        
        print(f"  Plotted {len(subtype_data)} animals")
    
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Remove empty subplots
for i in range(plot_idx + 1, 12):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(f'{output_dir}/actual_deviations_top12.png', dpi=300, bbox_inches='tight')
print(f"\n\nSaved: {output_dir}/actual_deviations_top12.png")
plt.close()

print("\n" + "="*70)
print("DONE!")
print("="*70)
