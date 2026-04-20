#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

print("="*70)
print("CREATING MARKER GENE VISUALIZATIONS")
print("="*70)

# ========== 1. CELL TYPE MARKER GENES - BAR PLOTS ==========
print("\nCreating cell type marker gene visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

# PCA components
pca_ct_files = ['pca_PC1_celltype_markers.csv', 'pca_PC3_celltype_markers.csv', 'pca_PC4_celltype_markers.csv']
pca_titles = ['PC1: Microglia', 'PC3: Astrocytes', 'PC4: GABAergic']

for idx, (file, title) in enumerate(zip(pca_ct_files, pca_titles)):
    ax = axes[idx]
    df = pd.read_csv(file)
    
    # Get top 15 from each direction
    pos = df[df['direction'] == 'positive'].head(15)
    neg = df[df['direction'] == 'negative'].head(15)
    
    # Combine and sort by absolute loading
    combined = pd.concat([pos, neg])
    combined['abs_loading'] = combined['loading'].abs()
    combined = combined.sort_values('abs_loading', ascending=True)
    
    # Plot
    colors = ['red' if x > 0 else 'blue' for x in combined['loading']]
    ax.barh(range(len(combined)), combined['loading'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['gene'], fontsize=7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Loading', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    # Add cell type labels
    pos_ct = pos.iloc[0]['cell_type']
    neg_ct = neg.iloc[0]['cell_type']
    ax.text(0.02, 0.98, f'Red: {pos_ct}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
    ax.text(0.02, 0.90, f'Blue: {neg_ct}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# FA factors
fa_ct_files = ['fa_Factor1_celltype_markers.csv', 'fa_Factor2_celltype_markers.csv', 'fa_Factor4_celltype_markers.csv']
fa_titles = ['Factor1: Microglia', 'Factor2: Astrocytes', 'Factor4: GABAergic']

for idx, (file, title) in enumerate(zip(fa_ct_files, fa_titles)):
    ax = axes[idx + 3]
    df = pd.read_csv(file)
    
    # Get top 15 from each direction
    pos = df[df['direction'] == 'positive'].head(15)
    neg = df[df['direction'] == 'negative'].head(15)
    
    # Combine and sort by absolute loading
    combined = pd.concat([pos, neg])
    combined['abs_loading'] = combined['loading'].abs()
    combined = combined.sort_values('abs_loading', ascending=True)
    
    # Plot
    colors = ['red' if x > 0 else 'blue' for x in combined['loading']]
    ax.barh(range(len(combined)), combined['loading'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['gene'], fontsize=7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Loading', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    # Add cell type labels
    pos_ct = pos.iloc[0]['cell_type']
    neg_ct = neg.iloc[0]['cell_type']
    ax.text(0.02, 0.98, f'Red: {pos_ct}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
    ax.text(0.02, 0.90, f'Blue: {neg_ct}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Cell Type Marker Genes by Component', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('celltype_marker_genes_visualization.png', dpi=300, bbox_inches='tight')
print("Saved: celltype_marker_genes_visualization.png")
plt.close()

# ========== 2. SUMMARY HEATMAP - TOP GENES ACROSS COMPONENTS ==========
print("\nCreating summary heatmap...")

# Load all cell type markers
all_ct_genes = set()
ct_gene_loadings = {}

for file in pca_ct_files + fa_ct_files:
    df = pd.read_csv(file)
    comp = file.split('_')[1] + '_' + file.split('_')[2].replace('celltype', '').replace('markers.csv', '')
    ct_gene_loadings[comp] = {}
    
    for idx, row in df.iterrows():
        gene = row['gene']
        all_ct_genes.add(gene)
        ct_gene_loadings[comp][gene] = row['loading']

# Get top genes by frequency across components
from collections import Counter
gene_freq = Counter()
for comp, genes_dict in ct_gene_loadings.items():
    for gene in list(genes_dict.keys())[:20]:  # Top 20 per component
        gene_freq[gene] += 1

# Get top 40 most frequent genes
top_genes = [gene for gene, count in gene_freq.most_common(40)]

# Create matrix
components = list(ct_gene_loadings.keys())
matrix = []
for gene in top_genes:
    row = []
    for comp in components:
        row.append(ct_gene_loadings[comp].get(gene, 0))
    matrix.append(row)

matrix_df = pd.DataFrame(matrix, index=top_genes, columns=components)

# Plot
fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(matrix_df, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Loading'},
            ax=ax, linewidths=0.5, yticklabels=True, xticklabels=True)
ax.set_xlabel('Component', fontweight='bold', fontsize=11)
ax.set_ylabel('Gene', fontweight='bold', fontsize=11)
ax.set_title('Top Cell Type Marker Genes Across All Components', 
             fontweight='bold', fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=8)
plt.tight_layout()
plt.savefig('celltype_markers_heatmap_summary.png', dpi=300, bbox_inches='tight')
print("Saved: celltype_markers_heatmap_summary.png")
plt.close()

print("\n" + "="*70)
print("Complete!")
print("="*70)
print("\nCreated:")
print("  1. celltype_marker_genes_visualization.png")
print("  2. celltype_markers_heatmap_summary.png")

