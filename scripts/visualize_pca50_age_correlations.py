#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load PCA 50 results
pca_df = pd.read_csv('pca_combined_all_celltypes_50pcs.csv')

print("Creating comprehensive age correlation heatmap for PCA (50 components)...")

# Create matrix: rows = cell_type-region, columns = PCs
# Values = age correlation (ALL, not just significant)

groups = []
pc_data = []

for cell_type in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = pca_df[pca_df['cell_type'] == cell_type]
    
    for region in sorted(ct_data['region'].unique()):
        region_data = ct_data[ct_data['region'] == region]
        
        if len(region_data) < 10:
            continue
        
        group_name = f"{cell_type}_{region}"
        groups.append(group_name)
        
        # Calculate correlation with age for ALL PCs (1-50)
        correlations = []
        for i in range(1, 51):
            pc = f'PC{i}'
            if pc in region_data.columns:
                r, p = stats.pearsonr(region_data['age'], region_data[pc])
                correlations.append(r)
            else:
                correlations.append(np.nan)
        
        pc_data.append(correlations)

# Create dataframe
heatmap_df = pd.DataFrame(
    pc_data,
    index=groups,
    columns=[f'PC{i}' for i in range(1, 51)]
)

# Sort by cell type for cleaner visualization
heatmap_df['cell_type'] = [g.split('_')[0] for g in groups]
heatmap_df['region'] = ['_'.join(g.split('_')[1:]) for g in groups]
heatmap_df = heatmap_df.sort_values(['cell_type', 'region'])
sorted_groups = [f"{row['cell_type']}_{row['region']}" for idx, row in heatmap_df.iterrows()]
heatmap_df = heatmap_df.drop(['cell_type', 'region'], axis=1)
heatmap_df.index = sorted_groups

print(f"\nHeatmap shape: {heatmap_df.shape}")
print(f"Groups: {len(groups)}")

# Create figure
fig, ax = plt.subplots(figsize=(18, 16))

# Plot heatmap
sns.heatmap(heatmap_df, 
            cmap='RdBu_r',
            center=0,
            vmin=-0.7, 
            vmax=0.7,
            cbar_kws={'label': 'Pearson Correlation with Age'},
            linewidths=0.1,
            linecolor='gray',
            ax=ax)

ax.set_title('Age Correlations Across All PCs (50 Components) and Cell Type/Region Combinations', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type - Region', fontsize=12, fontweight='bold')

# Adjust y-axis labels
plt.yticks(rotation=0, fontsize=8)
plt.xticks(rotation=90, fontsize=8)

# Add cell type separators
cell_type_breaks = []
current_ct = None
for i, group in enumerate(sorted_groups):
    ct = group.split('_')[0]
    if ct != current_ct:
        if current_ct is not None:
            cell_type_breaks.append(i)
        current_ct = ct

for break_idx in cell_type_breaks:
    ax.axhline(y=break_idx, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('pca50_age_correlation_heatmap_complete.png', dpi=300, bbox_inches='tight')
print("Saved: pca50_age_correlation_heatmap_complete.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

all_corrs = heatmap_df.values.flatten()
all_corrs = all_corrs[~np.isnan(all_corrs)]

print(f"\nTotal correlations: {len(all_corrs)}")
print(f"Mean |r|: {np.abs(all_corrs).mean():.3f}")
print(f"Median |r|: {np.median(np.abs(all_corrs)):.3f}")
print(f"Max |r|: {np.abs(all_corrs).max():.3f}")

# By PC
print("\n" + "="*70)
print("AGE CORRELATION STRENGTH BY PC")
print("="*70)

pc_stats = []
for pc in heatmap_df.columns:
    pc_corrs = heatmap_df[pc].values
    pc_corrs = pc_corrs[~np.isnan(pc_corrs)]
    
    if len(pc_corrs) > 0:
        mean_abs_r = np.abs(pc_corrs).mean()
        max_abs_r = np.abs(pc_corrs).max()
        
        # Count significant (rough estimate using |r| > 0.27 for n~50)
        n_strong = (np.abs(pc_corrs) > 0.27).sum()
        
        pc_stats.append({
            'PC': pc,
            'mean_|r|': mean_abs_r,
            'max_|r|': max_abs_r,
            'n_strong': n_strong
        })

pc_stats_df = pd.DataFrame(pc_stats)
pc_stats_df = pc_stats_df.sort_values('mean_|r|', ascending=False)

print("\nTop 10 PCs by mean |r|:")
print(pc_stats_df.head(10).to_string(index=False))

# By cell type
print("\n" + "="*70)
print("AGE CORRELATION STRENGTH BY CELL TYPE")
print("="*70)

for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_groups = [g for g in sorted_groups if g.startswith(ct)]
    ct_data = heatmap_df.loc[ct_groups]
    
    ct_values = ct_data.values.flatten()
    ct_values = ct_values[~np.isnan(ct_values)]
    
    mean_abs_r = np.abs(ct_values).mean()
    max_abs_r = np.abs(ct_values).max()
    
    print(f"\n{ct}:")
    print(f"  Mean |r| across all PCs/regions: {mean_abs_r:.3f}")
    print(f"  Max |r|: {max_abs_r:.3f}")

# Save the data
heatmap_df.to_csv('pca50_age_correlation_matrix_all.csv')
print("\nSaved: pca50_age_correlation_matrix_all.csv")

print("\n" + "="*70)
print("DIRECT COMPARISON: PCA-50 vs FA-50")
print("="*70)

# Load FA results for comparison
try:
    fa_heatmap = pd.read_csv('fa_age_correlation_matrix_all.csv', index_col=0)
    
    print(f"\nPCA-50 mean |r|: {np.abs(all_corrs).mean():.3f}")
    
    fa_corrs = fa_heatmap.values.flatten()
    fa_corrs = fa_corrs[~np.isnan(fa_corrs)]
    print(f"FA-50 mean |r|: {np.abs(fa_corrs).mean():.3f}")
    
    diff = np.abs(fa_corrs).mean() - np.abs(all_corrs).mean()
    pct_diff = (diff / np.abs(all_corrs).mean()) * 100
    
    print(f"\nDifference: {diff:.3f} ({pct_diff:+.1f}%)")
    
    if abs(pct_diff) < 5:
        print("→ PCA and FA are essentially EQUIVALENT")
    elif pct_diff > 5:
        print("→ FA is better")
    else:
        print("→ PCA is better")
        
except:
    print("\nFA results not loaded - run FA visualization first")

print("\n" + "="*70)
print("Complete!")
print("="*70)

