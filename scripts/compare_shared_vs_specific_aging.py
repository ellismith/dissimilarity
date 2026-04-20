#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load all significant PCA gene loadings
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_gene_loadings_*_PC*.csv')
print(f"Found {len(pca_files)} PCA gene loading files")

# Create gene sets for each cell type/region
gene_programs = {}

for file in pca_files:
    df = pd.read_csv(file)
    
    # Parse filename: pca_gene_loadings_CellType_Region_PC#.csv
    parts = file.split('/')[-1].replace('pca_gene_loadings_', '').replace('.csv', '').split('_')
    cell_type = parts[0]
    region = parts[1]
    pc = parts[2]
    
    key = f"{cell_type}_{region}"
    
    # Get top 50 genes
    top_genes = set(df.head(50)['gene_symbol'].dropna())
    gene_programs[key] = {
        'cell_type': cell_type,
        'region': region,
        'pc': pc,
        'genes': top_genes
    }

print(f"Loaded {len(gene_programs)} gene programs")

# === ANALYSIS 1: Within cell type, across regions ===
print("\n" + "="*70)
print("SHARED AGING PROGRAMS WITHIN CELL TYPES")
print("="*70)

cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']

for ct in cell_types:
    ct_programs = {k: v for k, v in gene_programs.items() if v['cell_type'] == ct}
    
    if len(ct_programs) < 2:
        continue
    
    print(f"\n{ct}:")
    print(f"  Regions analyzed: {len(ct_programs)}")
    
    # Pairwise overlaps
    keys = list(ct_programs.keys())
    overlaps = []
    
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            genes1 = ct_programs[keys[i]]['genes']
            genes2 = ct_programs[keys[j]]['genes']
            overlap = len(genes1 & genes2)
            overlaps.append(overlap)
            
            if overlap >= 10:  # Substantial overlap
                print(f"  {ct_programs[keys[i]]['region']} ↔ {ct_programs[keys[j]]['region']}: {overlap}/50 genes ({overlap/50*100:.0f}%)")
    
    if overlaps:
        print(f"  Mean overlap across regions: {np.mean(overlaps):.1f} genes ({np.mean(overlaps)/50*100:.0f}%)")
        if np.mean(overlaps) >= 10:
            print(f"  → SHARED aging program across regions")
        else:
            print(f"  → REGION-SPECIFIC aging programs")

# === ANALYSIS 2: Across cell types (same region) ===
print("\n" + "="*70)
print("SHARED AGING PROGRAMS ACROSS CELL TYPES")
print("="*70)

regions = set([v['region'] for v in gene_programs.values()])

for region in sorted(regions):
    region_programs = {k: v for k, v in gene_programs.items() if v['region'] == region}
    
    if len(region_programs) < 2:
        continue
    
    print(f"\n{region}:")
    print(f"  Cell types: {[v['cell_type'] for v in region_programs.values()]}")
    
    # Pairwise overlaps
    keys = list(region_programs.keys())
    overlaps = []
    
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            genes1 = region_programs[keys[i]]['genes']
            genes2 = region_programs[keys[j]]['genes']
            overlap = len(genes1 & genes2)
            overlaps.append(overlap)
            
            if overlap >= 5:  # Any overlap
                ct1 = region_programs[keys[i]]['cell_type']
                ct2 = region_programs[keys[j]]['cell_type']
                print(f"  {ct1} ↔ {ct2}: {overlap}/50 genes ({overlap/50*100:.0f}%)")
    
    if overlaps and np.mean(overlaps) >= 5:
        print(f"  → Some SHARED aging genes across cell types")
    elif overlaps:
        print(f"  → Mostly CELL-TYPE-SPECIFIC aging")

# === ANALYSIS 3: Universal aging genes ===
print("\n" + "="*70)
print("UNIVERSAL AGING GENES (found in 3+ programs)")
print("="*70)

from collections import Counter

all_genes = []
for program in gene_programs.values():
    all_genes.extend(list(program['genes']))

gene_counts = Counter(all_genes)
universal_genes = {gene: count for gene, count in gene_counts.items() if count >= 3}

if universal_genes:
    print(f"\nFound {len(universal_genes)} genes appearing in 3+ programs:\n")
    for gene, count in sorted(universal_genes.items(), key=lambda x: x[1], reverse=True)[:20]:
        # Find which cell types
        cell_types_with_gene = set()
        for program in gene_programs.values():
            if gene in program['genes']:
                cell_types_with_gene.add(program['cell_type'])
        print(f"  {gene}: {count} programs ({', '.join(sorted(cell_types_with_gene))})")
    
    print(f"\n→ These are UNIVERSAL aging genes!")
else:
    print("\nNo genes found in 3+ programs")
    print("→ Aging is HIGHLY CELL-TYPE/REGION-SPECIFIC")

# === Create heatmap ===
print("\n\nCreating overlap heatmap...")

# Create matrix of overlaps
program_names = list(gene_programs.keys())
n = len(program_names)
overlap_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            overlap_matrix[i, j] = 50
        else:
            genes_i = gene_programs[program_names[i]]['genes']
            genes_j = gene_programs[program_names[j]]['genes']
            overlap_matrix[i, j] = len(genes_i & genes_j)

# Plot
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(overlap_matrix, 
            xticklabels=program_names, 
            yticklabels=program_names,
            annot=True, 
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Gene Overlap (out of 50)'},
            ax=ax)
ax.set_title('Aging Gene Program Overlap Across Cell Types and Regions', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('aging_program_overlap_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: aging_program_overlap_heatmap.png")

print("\n=== Analysis complete! ===")

