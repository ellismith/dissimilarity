#!/usr/bin/env python

import pandas as pd

print("Loading gene loadings with symbols...")

# Load the CSV files (already have gene symbols as columns)
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)

print(f"PCA loadings shape: {pca_loadings.shape}")
print(f"FA loadings shape: {fa_loadings.shape}")

# Show top genes for various components
def show_top(loadings, component, n=20):
    row = loadings.loc[component]
    abs_loadings = row.abs().sort_values(ascending=False).head(n)
    
    print(f"\n{'='*70}")
    print(f"TOP {n} GENES FOR {component}")
    print(f"{'='*70}")
    print(f"{'Gene':<20} {'Loading':>12} {'|Loading|':>12}")
    print("-"*70)
    
    for gene in abs_loadings.index:
        loading_val = row.loc[gene]
        if isinstance(loading_val, pd.Series):
            loading_val = loading_val.iloc[0]
        loading = float(loading_val)
        print(f"{gene:<20} {loading:>12.6f} {abs(loading):>12.6f}")

# Just show examples of different PCs and Factors
print("\n" + "="*70)
print("EXAMPLES OF PCA GENE LOADINGS")
print("="*70)

show_top(pca_loadings, 'PC1', 20)
show_top(pca_loadings, 'PC10', 20)
show_top(pca_loadings, 'PC25', 20)

print("\n" + "="*70)
print("EXAMPLES OF FA GENE LOADINGS")
print("="*70)

show_top(fa_loadings, 'Factor1', 20)
show_top(fa_loadings, 'Factor10', 20)
show_top(fa_loadings, 'Factor25', 20)

# Compare overlap between corresponding dimensions
print(f"\n{'='*70}")
print("GENE OVERLAP: DO CORRESPONDING DIMENSIONS SHARE GENES?")
print(f"{'='*70}")

for i in [1, 10, 25]:
    pc_top = pca_loadings.loc[f'PC{i}'].abs().sort_values(ascending=False).head(50)
    factor_top = fa_loadings.loc[f'Factor{i}'].abs().sort_values(ascending=False).head(50)
    
    overlap = set(pc_top.index) & set(factor_top.index)
    print(f"\nPC{i} vs Factor{i}: {len(overlap)}/50 genes overlap ({len(overlap)*2}%)")
    
    if len(overlap) >= 5:
        print(f"  Some shared genes: {', '.join(list(overlap)[:5])}")

