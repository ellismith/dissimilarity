#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load gene loading files
gene_files = {
    'GABAergic_dlPFC_Factor10 (r=-0.65)': 'gene_loadings_GABAergic_dlPFC_Factor10_with_symbols.csv',
    'GABAergic_ACC_Factor7 (r=-0.54)': 'gene_loadings_GABAergic_ACC_Factor7_with_symbols.csv',
    'GABAergic_M1_Factor3 (r=-0.49)': 'gene_loadings_GABAergic_M1_Factor3_with_symbols.csv',
    'Astrocytes_CN_Factor7 (r=-0.58)': 'gene_loadings_Astrocytes_CN_Factor7_with_symbols.csv',
    'Astrocytes_M1_Factor8 (r=+0.52)': 'gene_loadings_Astrocytes_M1_Factor8_with_symbols.csv',
    'Astrocytes_ACC_Factor6 (r=-0.48)': 'gene_loadings_Astrocytes_ACC_Factor6_with_symbols.csv'
}

# Track age correlation direction
age_correlations = {
    'GABAergic_dlPFC_Factor10 (r=-0.65)': -0.651,
    'GABAergic_ACC_Factor7 (r=-0.54)': -0.538,
    'GABAergic_M1_Factor3 (r=-0.49)': -0.485,
    'Astrocytes_CN_Factor7 (r=-0.58)': -0.578,
    'Astrocytes_M1_Factor8 (r=+0.52)': 0.525,
    'Astrocytes_ACC_Factor6 (r=-0.48)': -0.483
}

# === 1. Bar plots showing top positive and negative loadings ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, file) in enumerate(gene_files.items()):
    ax = axes[idx]
    df = pd.read_csv(file)
    
    # Get top 10 positive and top 10 negative loadings
    top_pos = df.nlargest(10, 'loading')
    top_neg = df.nsmallest(10, 'loading')
    
    # Combine
    top_genes = pd.concat([top_pos, top_neg])
    
    # Remove genes without symbols
    top_genes = top_genes[~top_genes['gene_symbol'].isna()]
    top_genes = top_genes[~top_genes['gene_symbol'].str.startswith('ENSMMUG')]
    
    # Sort by loading
    top_genes = top_genes.sort_values('loading')
    
    # Shorten gene names if needed
    gene_labels = [g[:15] for g in top_genes['gene_symbol'].tolist()]
    loadings = top_genes['loading'].values
    
    # Color by sign
    colors = ['red' if l < 0 else 'blue' for l in loadings]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(gene_labels))
    bars = ax.barh(y_pos, loadings, color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=8)
    ax.set_xlabel('Factor Loading', fontsize=9, fontweight='bold')
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend for age interpretation
    age_corr = age_correlations[name]
    if age_corr < 0:
        # Negative age correlation: positive loading = decreases with age
        ax.text(0.02, 0.98, 'Blue = ↓ with age\nRed = ↑ with age', 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    else:
        # Positive age correlation: positive loading = increases with age
        ax.text(0.02, 0.98, 'Blue = ↑ with age\nRed = ↓ with age', 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('gene_loadings_top_genes.png', dpi=300, bbox_inches='tight')
print("Saved: gene_loadings_top_genes.png")

# === 2. Create separate lists of up vs down genes ===
print("\n" + "="*70)
print("GENES CHANGING WITH AGE - SEPARATED BY DIRECTION")
print("="*70)

summary_data = []

for name, file in gene_files.items():
    df = pd.read_csv(file)
    age_corr = age_correlations[name]
    
    # Remove genes without symbols
    df = df[~df['gene_symbol'].isna()]
    df = df[~df['gene_symbol'].str.startswith('ENSMMUG')]
    
    print(f"\n{name}")
    print("-" * 70)
    
    # Get top 15 positive and negative
    top_pos = df.nlargest(15, 'loading')
    top_neg = df.nsmallest(15, 'loading')
    
    if age_corr < 0:
        # Negative correlation: positive loadings = decrease with age
        print("  DECREASE with age (positive loadings):")
        print("    " + ", ".join(top_pos['gene_symbol'].head(10).tolist()))
        print("\n  INCREASE with age (negative loadings):")
        print("    " + ", ".join(top_neg['gene_symbol'].head(10).tolist()))
        
        summary_data.append({
            'analysis': name,
            'age_corr': age_corr,
            'genes_decrease': ', '.join(top_pos['gene_symbol'].head(20).tolist()),
            'genes_increase': ', '.join(top_neg['gene_symbol'].head(20).tolist())
        })
    else:
        # Positive correlation: positive loadings = increase with age
        print("  INCREASE with age (positive loadings):")
        print("    " + ", ".join(top_pos['gene_symbol'].head(10).tolist()))
        print("\n  DECREASE with age (negative loadings):")
        print("    " + ", ".join(top_neg['gene_symbol'].head(10).tolist()))
        
        summary_data.append({
            'analysis': name,
            'age_corr': age_corr,
            'genes_increase': ', '.join(top_pos['gene_symbol'].head(20).tolist()),
            'genes_decrease': ', '.join(top_neg['gene_symbol'].head(20).tolist())
        })

# Save summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('age_gene_direction_summary.csv', index=False)
print("\n" + "="*70)
print("Saved: age_gene_direction_summary.csv")

# === 3. Heatmap of top genes across all factors ===
print("\n" + "="*70)
print("Creating gene loading heatmap...")

# Collect top genes from each factor
all_top_genes = set()
for name, file in gene_files.items():
    df = pd.read_csv(file)
    df = df[~df['gene_symbol'].isna()]
    df = df[~df['gene_symbol'].str.startswith('ENSMMUG')]
    
    # Get top 15 by absolute loading
    top_genes = df.nlargest(15, 'abs_loading')['gene_symbol'].tolist()
    all_top_genes.update(top_genes)

print(f"Total unique top genes across all factors: {len(all_top_genes)}")

# Create matrix
gene_list = sorted(all_top_genes)
factor_names = list(gene_files.keys())

loading_matrix = pd.DataFrame(0.0, index=gene_list, columns=factor_names)

for name, file in gene_files.items():
    df = pd.read_csv(file)
    df = df[~df['gene_symbol'].isna()]
    df = df[~df['gene_symbol'].str.startswith('ENSMMUG')]
    
    # Handle duplicate gene symbols by keeping first occurrence
    df = df.drop_duplicates(subset='gene_symbol', keep='first')
    df = df.set_index('gene_symbol')
    
    for gene in gene_list:
        if gene in df.index:
            loading_matrix.loc[gene, name] = df.loc[gene, 'loading']

# Only plot genes that appear in at least one factor with |loading| > 0.3
gene_mask = (loading_matrix.abs() > 0.3).any(axis=1)
loading_matrix_filtered = loading_matrix[gene_mask]

print(f"Genes with |loading| > 0.3 in at least one factor: {len(loading_matrix_filtered)}")

if len(loading_matrix_filtered) > 0:
    fig, ax = plt.subplots(figsize=(14, max(8, len(loading_matrix_filtered)*0.25)))
    
    sns.heatmap(loading_matrix_filtered, 
                cmap='RdBu_r', 
                center=0,
                vmin=-0.8, vmax=0.8,
                cbar_kws={'label': 'Factor Loading'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    ax.set_xlabel('Factor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
    ax.set_title('Gene Loadings Across Age-Associated Factors', fontsize=13, fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('gene_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: gene_loadings_heatmap.png")
else:
    print("Not enough genes with strong loadings to create heatmap")

print("\n" + "="*70)
print("Visualization complete!")
print("="*70)

