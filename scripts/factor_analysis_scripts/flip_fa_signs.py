#!/usr/bin/env python

import pandas as pd
import numpy as np

print("="*70)
print("FLIPPING FA SIGNS TO MATCH PCA CONVENTION")
print("="*70)

# Load FA loadings and scores
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)
fa_combined = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

# Flip signs for all factors
print("\nFlipping signs for all 50 factors...")
for i in range(1, 51):
    factor_name = f'Factor{i}'
    if factor_name in fa_loadings.index:
        fa_loadings.loc[factor_name] = -fa_loadings.loc[factor_name]
    if factor_name in fa_combined.columns:
        fa_combined[factor_name] = -fa_combined[factor_name]

# Save flipped versions
fa_loadings.to_csv('fa_all_loadings_with_symbols.csv')
fa_combined.to_csv('fa_combined_all_celltypes_50factors.csv', index=False)

print("Flipped and saved!")
print("\nNow rerun:")
print("  1. python3 analyze_all_50_components.py")
print("  2. python3 extract_celltype_marker_genes.py")
print("  3. python3 extract_region_marker_genes.py")
print("  4. python3 visualize_marker_genes.py")
print("  5. python3 visualize_all_region_markers.py")

