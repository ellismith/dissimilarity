#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load FA results
fa_df = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

print("Creating comprehensive age correlation heatmap for Factor Analysis...")

# Create matrix: rows = cell_type-region, columns = Factors
# Values = age correlation (ALL, not just significant)

groups = []
factor_data = []

for cell_type in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = fa_df[fa_df['cell_type'] == cell_type]
    
    for region in sorted(ct_data['region'].unique()):
        region_data = ct_data[ct_data['region'] == region]
        
        if len(region_data) < 10:
            continue
        
        group_name = f"{cell_type}_{region}"
        groups.append(group_name)
        
        # Calculate correlation with age for ALL Factors
        correlations = []
        for i in range(1, 51):
            factor = f'Factor{i}'
            if factor in region_data.columns:
                r, p = stats.pearsonr(region_data['age'], region_data[factor])
                correlations.append(r)
            else:
                correlations.append(np.nan)
        
        factor_data.append(correlations)

# Create dataframe
heatmap_df = pd.DataFrame(
    factor_data,
    index=groups,
    columns=[f'Factor{i}' for i in range(1, 51)]
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

ax.set_title('Age Correlations Across All Factors (Factor Analysis) and Cell Type/Region Combinations', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Factor', fontsize=12, fontweight='bold')
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
plt.savefig('fa_age_correlation_heatmap_complete.png', dpi=300, bbox_inches='tight')
print("Saved: fa_age_correlation_heatmap_complete.png")

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

# By Factor
print("\n" + "="*70)
print("AGE CORRELATION STRENGTH BY FACTOR")
print("="*70)

factor_stats = []
for factor in heatmap_df.columns:
    factor_corrs = heatmap_df[factor].values
    factor_corrs = factor_corrs[~np.isnan(factor_corrs)]
    
    if len(factor_corrs) > 0:
        mean_abs_r = np.abs(factor_corrs).mean()
        max_abs_r = np.abs(factor_corrs).max()
        
        # Count significant (rough estimate using |r| > 0.27 for n~50)
        n_strong = (np.abs(factor_corrs) > 0.27).sum()
        
        factor_stats.append({
            'Factor': factor,
            'mean_|r|': mean_abs_r,
            'max_|r|': max_abs_r,
            'n_strong': n_strong
        })

factor_stats_df = pd.DataFrame(factor_stats)
factor_stats_df = factor_stats_df.sort_values('mean_|r|', ascending=False)

print("\nTop 10 Factors by mean |r|:")
print(factor_stats_df.head(10).to_string(index=False))

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
    print(f"  Mean |r| across all Factors/regions: {mean_abs_r:.3f}")
    print(f"  Max |r|: {max_abs_r:.3f}")

# Save the data
heatmap_df.to_csv('fa_age_correlation_matrix_all.csv')
print("\nSaved: fa_age_correlation_matrix_all.csv")

print("\n" + "="*70)
print("Complete!")
print("="*70)

