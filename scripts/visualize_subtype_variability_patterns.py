#!/usr/bin/env python
"""
Visualize subtype-level variability patterns with proper ordering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

print("="*70)
print("VISUALIZING SUBTYPE VARIABILITY PATTERNS")
print("="*70)

# Load all subtype variability results
results_dir = "/scratch/easmit31/factor_analysis/subtype_variability"
result_files = glob.glob(f"{results_dir}/subtype_variability_*.csv")

print(f"\nFound {len(result_files)} result files")

all_results = []
for f in result_files:
    df = pd.read_csv(f)
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
print(f"Total results: {len(combined)} subtype-PC-region combinations")

# Focus on Glutamatergic and GABAergic
glut_gaba = combined[combined['cell_type'].isin(['Glutamatergic', 'GABAergic'])].copy()
print(f"Glutamatergic + GABAergic: {len(glut_gaba)} combinations")

# Extract numeric part of subtype for sorting
def extract_subtype_number(subtype_name):
    """Extract the number from subtype name for proper sorting"""
    match = re.search(r'_(\d+)$', subtype_name)
    if match:
        return int(match.group(1))
    return -1

glut_gaba['subtype_num'] = glut_gaba['subtype'].apply(extract_subtype_number)

output_dir = "/scratch/easmit31/factor_analysis/subtype_variability/figures"

# For each subtype, get the strongest correlation across all PCs
print("\nGetting strongest correlation per subtype...")
subtype_best = glut_gaba.loc[glut_gaba.groupby(['cell_type', 'region', 'subtype'])['abs_dev_corr'].apply(lambda x: x.abs().idxmax())]

# Figure 1: Heatmap showing strongest correlation per subtype-region (ORDERED)
print("\nCreating Figure 1: Subtype variability heatmaps by cell type (ordered)")

for cell_type in ['Glutamatergic', 'GABAergic']:
    data = subtype_best[subtype_best['cell_type'] == cell_type].copy()
    
    if len(data) == 0:
        continue
    
    # Pivot to get subtype x region
    pivot = data.pivot_table(index='subtype', columns='region', values='abs_dev_corr')
    
    # Extract subtype numbers for sorting
    subtype_order = sorted(pivot.index, key=extract_subtype_number)
    pivot = pivot.reindex(subtype_order)
    
    # Only show subtypes present in at least 3 regions
    pivot = pivot[pivot.notna().sum(axis=1) >= 3]
    
    if len(pivot) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.3)))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Strongest Correlation (r)'}, ax=ax,
                vmin=-0.6, vmax=0.6)
    ax.set_title(f'{cell_type} Subtypes\nStrongest Age-Variability Correlation per Subtype-Region',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Subtype', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{cell_type}_subtype_variability_heatmap_ordered.png",
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{cell_type}_subtype_variability_heatmap_ordered.png")
    plt.close()

# Figure 2: Which PC shows the strongest effect per subtype? (FIXED ORDERING)
print("\nCreating Figure 2: Which PC drives variability per subtype")

# Count which PC has strongest effect across subtypes
pc_counts = subtype_best.groupby(['cell_type', 'PC']).size().reset_index(name='count')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, cell_type in enumerate(['Glutamatergic', 'GABAergic']):
    data = pc_counts[pc_counts['cell_type'] == cell_type]
    
    # Sort PCs numerically
    pc_order = sorted(data['PC'].unique(), key=lambda x: int(x.replace('PC', '')))
    
    # Create properly ordered data
    counts = []
    for pc in pc_order:
        count_val = data[data['PC'] == pc]['count'].values
        counts.append(count_val[0] if len(count_val) > 0 else 0)
    
    x_pos = np.arange(len(pc_order))
    axes[idx].bar(x_pos, counts, color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].set_xticks(x_pos)
    axes[idx].set_xticklabels(pc_order, rotation=0)
    axes[idx].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Number of Subtypes', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{cell_type}\nWhich PC Shows Strongest Variability Change?',
                       fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/subtype_which_pc_drives_ordered.png",
            dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/subtype_which_pc_drives_ordered.png")
plt.close()

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for cell_type in ['Glutamatergic', 'GABAergic']:
    data = subtype_best[subtype_best['cell_type'] == cell_type]
    
    print(f"\n{cell_type}:")
    print(f"  Total subtypes: {len(data)}")
    print(f"  Increasing variability (r>0): {(data['abs_dev_corr'] > 0).sum()}")
    print(f"  Decreasing variability (r<0): {(data['abs_dev_corr'] < 0).sum()}")
    print(f"  Mean correlation: {data['abs_dev_corr'].mean():.3f}")
    print(f"  Median correlation: {data['abs_dev_corr'].median():.3f}")
    print(f"  Significant (FDR<0.05): {(data['abs_dev_fdr'] < 0.05).sum()}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
