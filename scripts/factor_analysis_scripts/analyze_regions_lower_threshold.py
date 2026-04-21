#!/usr/bin/env python

import pandas as pd
import numpy as np

print("="*70)
print("ANALYZING REGION COMPONENTS WITH η² > 0.1")
print("="*70)

# Load component analysis
pca_analysis = pd.read_csv('pca_all_50_components_analyzed.csv')
fa_analysis = pd.read_csv('fa_all_50_factors_analyzed.csv')

# Get region components with lower threshold
pca_reg_comps = pca_analysis[pca_analysis['Region_eta2'] > 0.1]['Component'].tolist()
fa_reg_comps = fa_analysis[fa_analysis['Region_eta2'] > 0.1]['Component'].tolist()

print(f"\nPCA: {len(pca_reg_comps)} region components (η² > 0.1)")
print(f"FA: {len(fa_reg_comps)} region factors (η² > 0.1)")

print("\nPCA components:")
for comp in pca_reg_comps:
    row = pca_analysis[pca_analysis['Component'] == comp].iloc[0]
    print(f"  {comp}: η²={row['Region_eta2']:.2f}, Dominant={row['Dominant_Region']}")

print("\nFA factors:")
for comp in fa_reg_comps:
    row = fa_analysis[fa_analysis['Component'] == comp].iloc[0]
    print(f"  {comp}: η²={row['Region_eta2']:.2f}, Dominant={row['Dominant_Region']}")

# Save lists
pca_reg_df = pca_analysis[pca_analysis['Region_eta2'] > 0.1][['Component', 'Region_eta2', 'Dominant_Region', 'Top3_Regions']]
pca_reg_df.to_csv('pca_region_components_0.1.csv', index=False)

fa_reg_df = fa_analysis[fa_analysis['Region_eta2'] > 0.1][['Component', 'Region_eta2', 'Dominant_Region', 'Top3_Regions']]
fa_reg_df.to_csv('fa_region_components_0.1.csv', index=False)

print("\nSaved: pca_region_components_0.1.csv")
print("Saved: fa_region_components_0.1.csv")

# ========== EXTRACT MARKER GENES FOR NEW COMPONENTS ==========
print("\n" + "="*70)
print("EXTRACTING MARKER GENES FOR NEW REGION COMPONENTS")
print("="*70)

# Load loadings
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)

def extract_marker_genes(loadings_df, component, top_n=50):
    comp_loadings = loadings_df.loc[component]
    positive_genes = comp_loadings[comp_loadings > 0].sort_values(ascending=False).head(top_n)
    negative_genes = comp_loadings[comp_loadings < 0].sort_values(ascending=True).head(top_n)
    return positive_genes, negative_genes

regions = ['ACC', 'NAc', 'HIP', 'CN', 'mdTN', 'dlPFC', 'EC', 'IPP', 'M1', 'lCb', 'MB']

# Extract for PCA
for pc in pca_reg_comps:
    pc_row = pca_analysis[pca_analysis['Component'] == pc].iloc[0]
    pos_genes, neg_genes = extract_marker_genes(pca_loadings, pc)
    
    # Get region scores
    region_scores = {reg: pc_row[f'Reg_{reg}'] for reg in regions}
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1])
    
    most_negative_regions = [reg for reg, score in sorted_regions[:3]]
    most_positive_regions = [reg for reg, score in sorted_regions[-3:]]
    
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'regions': ', '.join(most_negative_regions),
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'regions': ', '.join(most_positive_regions),
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'pca_{pc}_region_markers.csv', index=False)

print(f"Extracted marker genes for {len(pca_reg_comps)} PCA components")

# Extract for FA
for factor in fa_reg_comps:
    fa_row = fa_analysis[fa_analysis['Component'] == factor].iloc[0]
    pos_genes, neg_genes = extract_marker_genes(fa_loadings, factor)
    
    # Get region scores
    region_scores = {reg: fa_row[f'Reg_{reg}'] for reg in regions}
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1])
    
    most_negative_regions = [reg for reg, score in sorted_regions[:3]]
    most_positive_regions = [reg for reg, score in sorted_regions[-3:]]
    
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'regions': ', '.join(most_negative_regions),
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'regions': ', '.join(most_positive_regions),
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'fa_{factor}_region_markers.csv', index=False)

print(f"Extracted marker genes for {len(fa_reg_comps)} FA factors")

print("\n" + "="*70)
print("Complete!")
print("="*70)

