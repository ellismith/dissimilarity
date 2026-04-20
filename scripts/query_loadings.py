#!/usr/bin/env python

import pandas as pd
import sys

# Load loadings
print("Loading loadings...", flush=True)
pca_loadings = pd.read_csv('pca_all_loadings.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings.csv', index_col=0)

# Get gene symbols (map from Ensembl IDs)
# For now, we'll work with Ensembl IDs

def show_top_genes(component, n=20):
    """Show top N genes for a PC or Factor"""
    
    if component.startswith('PC'):
        loadings = pca_loadings.loc[component]
    elif component.startswith('Factor'):
        loadings = fa_loadings.loc[component]
    else:
        print(f"Unknown component: {component}")
        return
    
    abs_loadings = loadings.abs().sort_values(ascending=False)
    
    print(f"\n{'='*70}")
    print(f"TOP {n} GENES FOR {component}")
    print(f"{'='*70}")
    print(f"{'Gene':<25} {'Loading':>12} {'|Loading|':>12}")
    print("-"*70)
    
    for gene in abs_loadings.head(n).index:
        loading = loadings[gene]
        abs_load = abs(loading)
        print(f"{gene:<25} {loading:>12.6f} {abs_load:>12.6f}")

# Examples
show_top_genes('PC19', 20)
show_top_genes('PC20', 20)
show_top_genes('Factor26', 20)
show_top_genes('Factor42', 20)
show_top_genes('Factor48', 20)

# Compare PC20 vs Factor26 genes
print(f"\n{'='*70}")
print("COMPARING PC20 vs FACTOR26 TOP GENES")
print(f"{'='*70}")

pc20_top = pca_loadings.loc['PC20'].abs().sort_values(ascending=False).head(50)
f26_top = fa_loadings.loc['Factor26'].abs().sort_values(ascending=False).head(50)

overlap = set(pc20_top.index) & set(f26_top.index)

print(f"\nOverlap: {len(overlap)}/50 genes ({len(overlap)*2}%)")
if len(overlap) > 0:
    print(f"\nShared genes:")
    for gene in list(overlap)[:20]:
        pc20_load = pca_loadings.loc['PC20', gene]
        f26_load = fa_loadings.loc['Factor26', gene]
        print(f"  {gene}: PC20={pc20_load:>8.4f}, Factor26={f26_load:>8.4f}")

