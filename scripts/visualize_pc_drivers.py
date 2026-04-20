#!/usr/bin/env python
"""
Visualize what drives each PC across cell types and regions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading PC drivers data...")
df = pd.read_csv('/scratch/easmit31/factor_analysis/pc_drivers_summary.csv')

output_dir = '/scratch/easmit31/factor_analysis/subtype_variability/figures'

# Figure 1: Variance explained by each PC
print("\nCreating Figure 1: Variance explained by PC")
fig, ax = plt.subplots(figsize=(10, 6))

pc_order = [f'PC{i}' for i in range(1, 11)]
var_data = df.groupby('PC')['variance_explained'].agg(['mean', 'std'])
var_data = var_data.reindex(pc_order)

ax.bar(range(len(var_data)), var_data['mean'], yerr=var_data['std'], 
       capsize=5, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(var_data)))
ax.set_xticklabels(var_data.index)
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Variance Explained', fontsize=12, fontweight='bold')
ax.set_title('Average Variance Explained by Each PC\n(across all cell type-region combinations)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/pc_variance_explained.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/pc_variance_explained.png")
plt.close()

# Figure 2: Heatmap of age correlations
print("\nCreating Figure 2: Age correlation heatmap")
pivot_age = df.pivot_table(index=['cell_type', 'region'], columns='PC', values='age_corr')
pivot_age = pivot_age[pc_order]  # Reorder columns

fig, ax = plt.subplots(figsize=(12, 18))
sns.heatmap(pivot_age, cmap='RdBu_r', center=0, annot=False, 
            cbar_kws={'label': 'Spearman r'}, ax=ax, vmin=-0.8, vmax=0.8)
ax.set_title('Age Correlations Across All PCs\n(cell type-region combinations)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Cell Type - Region', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/pc_age_correlations_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/pc_age_correlations_heatmap.png")
plt.close()

# Figure 3: Which PC captures age the strongest per cell type-region?
print("\nCreating Figure 3: Strongest age-associated PC per combo")

# Get strongest absolute age correlation per cell type-region
strongest_age = df.loc[df.groupby(['cell_type', 'region'])['age_corr'].apply(lambda x: x.abs().idxmax())]
strongest_age = strongest_age.sort_values('age_corr', key=abs, ascending=False).head(30)

fig, ax = plt.subplots(figsize=(12, 10))
colors = ['red' if x > 0 else 'blue' for x in strongest_age['age_corr']]
y_labels = [f"{row['cell_type']}-{row['region']}-{row['PC']}" 
            for _, row in strongest_age.iterrows()]

ax.barh(range(len(strongest_age)), strongest_age['age_corr'], color=colors, alpha=0.7)
ax.set_yticks(range(len(strongest_age)))
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Age Correlation (r)', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Strongest Age-Associated PCs\n(one PC per cell type-region)', 
             fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

# Add significance markers
for i, (_, row) in enumerate(strongest_age.iterrows()):
    if row['age_pval'] < 0.001:
        marker = '***'
    elif row['age_pval'] < 0.01:
        marker = '**'
    elif row['age_pval'] < 0.05:
        marker = '*'
    else:
        marker = ''
    
    if marker:
        ax.text(row['age_corr'], i, f' {marker}', 
               ha='left' if row['age_corr'] > 0 else 'right',
               va='center', fontsize=10, fontweight='bold')

ax.legend(handles=[
    plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, label='Positive'),
    plt.Rectangle((0,0),1,1, fc='blue', alpha=0.7, label='Negative')
], loc='lower right')

plt.tight_layout()
plt.savefig(f'{output_dir}/pc_strongest_age_associations.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/pc_strongest_age_associations.png")
plt.close()

# Figure 4: Distribution of age correlations by PC
print("\nCreating Figure 4: Age correlation distributions by PC")
fig, ax = plt.subplots(figsize=(12, 6))

pc_data = [df[df['PC'] == pc]['age_corr'].values for pc in pc_order]
bp = ax.boxplot(pc_data, labels=pc_order, patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Age Correlation (r)', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Age Correlations by PC\n(across all cell type-region combinations)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/pc_age_correlation_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/pc_age_correlation_distributions.png")
plt.close()

# Figure 5: Cell type comparison
print("\nCreating Figure 5: Age associations by cell type")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, cell_type in enumerate(['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']):
    ct_data = df[df['cell_type'] == cell_type]
    
    if len(ct_data) == 0:
        continue
    
    # Heatmap of age correlations
    pivot = ct_data.pivot_table(index='region', columns='PC', values='age_corr')
    pivot = pivot[pc_order]  # Reorder
    
    sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Spearman r'}, ax=axes[idx], 
                vmin=-0.6, vmax=0.6)
    axes[idx].set_title(f'{cell_type}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('PC', fontsize=10)
    axes[idx].set_ylabel('Region', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/pc_age_by_celltype.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/pc_age_by_celltype.png")
plt.close()

print("\n" + "="*70)
print("ALL FIGURES SAVED!")
print("="*70)
