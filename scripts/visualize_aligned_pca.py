#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load results
pca_df = pd.read_csv('pca_combined_all_celltypes.csv')
results_df = pd.read_csv('aligned_pca_age_results.csv')
sig_results = results_df[results_df['is_significant']]

print("Creating comprehensive visualizations...")

# === 1. HEATMAP: Which PCs are age-associated in which groups ===
print("\n1. Creating PC × Group heatmap...")

# Create matrix: rows=groups, columns=PCs, values=age correlation
groups = []
pc_corrs = {f'PC{i}': [] for i in range(1, 21)}

for idx, row in results_df.iterrows():
    group = f"{row['cell_type']}\n{row['region']}"
    groups.append(group)
    
    # Get correlations for each PC
    ct_region_data = pca_df[(pca_df['cell_type'] == row['cell_type']) & 
                            (pca_df['region'] == row['region'])]
    
    for i in range(1, 21):
        pc = f'PC{i}'
        if len(ct_region_data) >= 10:
            r, p = stats.pearsonr(ct_region_data['age'], ct_region_data[pc])
            if p < 0.05:
                pc_corrs[pc].append(r)
            else:
                pc_corrs[pc].append(0)  # Non-sig = 0
        else:
            pc_corrs[pc].append(np.nan)

# Create dataframe
heatmap_data = pd.DataFrame(pc_corrs, index=groups)

# Plot
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(heatmap_data, 
            cmap='RdBu_r',
            center=0,
            vmin=-0.7, vmax=0.7,
            cbar_kws={'label': 'Age Correlation (r)'},
            linewidths=0.1,
            ax=ax)
ax.set_title('Age Correlations Across All PCs and Groups\n(Only significant shown, non-sig = 0)', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type - Region', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig('pc_age_heatmap_all.png', dpi=300, bbox_inches='tight')
print("Saved: pc_age_heatmap_all.png")

# === 2. BAR PLOT: How many groups does each PC affect? ===
print("\n2. Creating PC frequency bar plot...")

pc_counts = {}
for i in range(1, 21):
    pc = f'PC{i}'
    n_groups = (sig_results['best_pc'] == pc).sum()
    pc_counts[pc] = n_groups

fig, ax = plt.subplots(figsize=(12, 6))
pcs = list(pc_counts.keys())
counts = list(pc_counts.values())

colors = ['red' if c >= 4 else 'gray' for c in counts]
bars = ax.bar(pcs, counts, color=colors, alpha=0.7, edgecolor='black')

# Highlight key PCs
for i, (pc, count) in enumerate(pc_counts.items()):
    if count >= 4:
        ax.text(i, count + 0.3, str(count), ha='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Groups with Significant Age Effect', fontsize=12, fontweight='bold')
ax.set_title('Which PCs Capture Aging Most Frequently?\n(Red = ≥4 groups)', 
            fontsize=13, fontweight='bold')
ax.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Threshold (4 groups)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pc_frequency_barplot.png', dpi=300, bbox_inches='tight')
print("Saved: pc_frequency_barplot.png")

# === 3. CELL TYPE BREAKDOWN: Which PCs for which cell types? ===
print("\n3. Creating cell type × PC heatmap...")

cell_type_pc = {}
for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_sig = sig_results[sig_results['cell_type'] == ct]
    pc_dist = ct_sig['best_pc'].value_counts()
    cell_type_pc[ct] = pc_dist

# Create matrix
ct_pc_df = pd.DataFrame(cell_type_pc).fillna(0).T

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(ct_pc_df, 
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Regions'},
            linewidths=0.5,
            ax=ax)
ax.set_title('PC Usage by Cell Type\n(Which PCs capture aging in which cell types?)', 
            fontsize=13, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('celltype_pc_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: celltype_pc_heatmap.png")

# === 4. SCATTER: Age correlation strength vs PC number ===
print("\n4. Creating PC number vs correlation scatter...")

pc_nums = []
correlations = []
cell_types_list = []

for idx, row in sig_results.iterrows():
    pc_num = int(row['best_pc'].replace('PC', ''))
    pc_nums.append(pc_num)
    correlations.append(abs(row['best_r']))
    cell_types_list.append(row['cell_type'])

fig, ax = plt.subplots(figsize=(12, 6))

colors_map = {'Glutamatergic': 'blue', 'GABAergic': 'green', 
              'Astrocytes': 'orange', 'Microglia': 'red'}
for ct in colors_map:
    mask = np.array(cell_types_list) == ct
    ax.scatter(np.array(pc_nums)[mask], np.array(correlations)[mask], 
              label=ct, color=colors_map[ct], s=100, alpha=0.7, edgecolors='black')

ax.set_xlabel('PC Number', fontsize=12, fontweight='bold')
ax.set_ylabel('|Age Correlation|', fontsize=12, fontweight='bold')
ax.set_title('Age Signal Strength vs PC Number\n(Later PCs often capture aging)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pc_number_vs_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: pc_number_vs_correlation.png")

# === 5. SUMMARY TABLE ===
print("\n5. Creating summary table...")

summary = pd.DataFrame({
    'PC': [f'PC{i}' for i in range(1, 21)],
    'N_groups': [pc_counts[f'PC{i}'] for i in range(1, 21)],
    'Mean_|r|': [heatmap_data[f'PC{i}'].abs().mean() for i in range(1, 21)],
    'Max_|r|': [heatmap_data[f'PC{i}'].abs().max() for i in range(1, 21)]
})

summary = summary[summary['N_groups'] > 0].sort_values('N_groups', ascending=False)

print("\nTop PCs by frequency:")
print(summary.head(10).to_string(index=False))

summary.to_csv('pc_aging_summary.csv', index=False)
print("\nSaved: pc_aging_summary.csv")

print("\n" + "="*70)
print("All visualizations complete!")
print("="*70)

