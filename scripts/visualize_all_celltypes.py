#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the summary
sig_results = pd.read_csv('factor_analysis_all_celltypes_significant.csv')

print(f"Loaded {len(sig_results)} significant age-associated factors")

# === 1. Heatmap: Age correlations by cell type and region ===
print("\nCreating heatmap of age correlations...")

# Create pivot table
pivot_data = sig_results.pivot_table(
    values='best_age_corr',
    index='region',
    columns='cell_type',
    aggfunc='first'
)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(pivot_data, 
            cmap='RdBu_r', 
            center=0,
            vmin=-0.8, vmax=0.8,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Age Correlation (r)'},
            linewidths=0.5,
            linecolor='white',
            ax=ax)

ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_title('Age-Associated Factor Correlations Across Cell Types and Regions', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('age_correlations_heatmap_all_celltypes.png', dpi=300, bbox_inches='tight')
print("Saved: age_correlations_heatmap_all_celltypes.png")

# === 2. Bar plot: Strongest signals per cell type ===
print("\nCreating bar plot of top signals...")

fig, ax = plt.subplots(figsize=(14, 8))

# Get top 3 per cell type
top_per_celltype = []
for ct in sig_results['cell_type'].unique():
    ct_data = sig_results[sig_results['cell_type'] == ct].copy()
    ct_data['abs_corr'] = ct_data['best_age_corr'].abs()
    ct_data = ct_data.nlargest(3, 'abs_corr')
    top_per_celltype.append(ct_data)

top_data = pd.concat(top_per_celltype)
top_data['label'] = top_data['cell_type'] + '\n' + top_data['region']

# Sort by absolute correlation
top_data['abs_corr'] = top_data['best_age_corr'].abs()
top_data = top_data.sort_values('abs_corr', ascending=True)

colors = ['red' if x < 0 else 'blue' for x in top_data['best_age_corr']]

bars = ax.barh(range(len(top_data)), top_data['best_age_corr'], color=colors, alpha=0.7)

ax.set_yticks(range(len(top_data)))
ax.set_yticklabels(top_data['label'], fontsize=9)
ax.set_xlabel('Age Correlation (r)', fontsize=11, fontweight='bold')
ax.set_title('Top 3 Age-Associated Factors per Cell Type', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('top_signals_by_celltype.png', dpi=300, bbox_inches='tight')
print("Saved: top_signals_by_celltype.png")

# === 3. Violin plot: Distribution of age correlations by cell type ===
print("\nCreating violin plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Add absolute correlation column for ordering
cell_order_data = sig_results.groupby('cell_type')['best_age_corr'].apply(lambda x: x.abs().mean())
cell_order = cell_order_data.sort_values(ascending=False).index

sns.violinplot(data=sig_results, x='cell_type', y='best_age_corr', 
              order=cell_order, palette='Set2', ax=ax)

# Add individual points
sns.stripplot(data=sig_results, x='cell_type', y='best_age_corr',
             order=cell_order, color='black', size=4, alpha=0.5, ax=ax)

ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
ax.set_ylabel('Age Correlation (r)', fontsize=11, fontweight='bold')
ax.set_title('Distribution of Age Effects Across Cell Types', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('age_correlation_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: age_correlation_distributions.png")

# === 4. Regional comparison across cell types ===
print("\nCreating regional comparison...")

# Calculate mean |correlation| per region across cell types
region_summary = sig_results.groupby('region').agg({
    'best_age_corr': lambda x: x.abs().mean(),
    'cell_type': 'count'
}).rename(columns={'best_age_corr': 'mean_abs_corr', 'cell_type': 'n_celltypes'})
region_summary = region_summary.sort_values('mean_abs_corr', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Mean absolute correlation
ax = axes[0]
bars = ax.barh(range(len(region_summary)), region_summary['mean_abs_corr'], 
              color='steelblue', alpha=0.7)
ax.set_yticks(range(len(region_summary)))
ax.set_yticklabels(region_summary.index)
ax.set_xlabel('Mean |Age Correlation|', fontsize=10, fontweight='bold')
ax.set_title('Brain Regions Ranked by\nAge Effect Strength', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Plot 2: Number of cell types with significant effects
ax = axes[1]
bars = ax.barh(range(len(region_summary)), region_summary['n_celltypes'],
              color='coral', alpha=0.7)
ax.set_yticks(range(len(region_summary)))
ax.set_yticklabels(region_summary.index)
ax.set_xlabel('Number of Cell Types', fontsize=10, fontweight='bold')
ax.set_title('Brain Regions Ranked by\nNumber of Aging Cell Types', fontsize=11, fontweight='bold')
ax.set_xticks(range(5))
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('regional_aging_summary.png', dpi=300, bbox_inches='tight')
print("Saved: regional_aging_summary.png")

# === 5. Statistical summary table ===
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print("\nBy Cell Type:")
ct_summary = sig_results.groupby('cell_type').agg({
    'best_age_corr': ['count', lambda x: x.abs().mean(), 'min', 'max'],
    'best_pval': lambda x: (x < 0.001).sum()
}).round(3)
ct_summary.columns = ['N_regions', 'Mean_|r|', 'Min_r', 'Max_r', 'N_p<0.001']
print(ct_summary.to_string())

print("\nBy Region:")
region_summary_full = sig_results.groupby('region').agg({
    'best_age_corr': ['count', lambda x: x.abs().mean(), 'min', 'max'],
}).round(3)
region_summary_full.columns = ['N_celltypes', 'Mean_|r|', 'Min_r', 'Max_r']
region_summary_full = region_summary_full.sort_values('Mean_|r|', ascending=False)
print(region_summary_full.to_string())

# === 6. Direction of effects ===
print("\n" + "="*70)
print("DIRECTION OF AGE EFFECTS")
print("="*70)

for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = sig_results[sig_results['cell_type'] == ct]
    n_pos = (ct_data['best_age_corr'] > 0).sum()
    n_neg = (ct_data['best_age_corr'] < 0).sum()
    print(f"\n{ct}:")
    print(f"  Negative correlations (factor decreases with age): {n_neg}")
    print(f"  Positive correlations (factor increases with age): {n_pos}")

print("\n" + "="*70)
print("Visualization complete!")
print("="*70)

