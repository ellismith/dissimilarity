#!/usr/bin/env python

import pandas as pd
import numpy as np

print("="*70)
print("EXTRACTING REGION MARKER GENES - SEPARATE LOGIC FOR PCA VS FA")
print("="*70)

# Load loadings
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)

# Load component analysis
pca_analysis = pd.read_csv('pca_all_50_components_analyzed.csv')
fa_analysis = pd.read_csv('fa_all_50_factors_analyzed.csv')

# Get region dominated components (using 0.1 threshold)
pca_reg_comps = pca_analysis[pca_analysis['Region_eta2'] > 0.1]['Component'].tolist()
fa_reg_comps = fa_analysis[fa_analysis['Region_eta2'] > 0.1]['Component'].tolist()

print(f"\nAnalyzing {len(pca_reg_comps)} PCA components and {len(fa_reg_comps)} FA factors")

def extract_marker_genes(loadings_df, component, top_n=50):
    comp_loadings = loadings_df.loc[component]
    positive_genes = comp_loadings[comp_loadings > 0].sort_values(ascending=False).head(top_n)
    negative_genes = comp_loadings[comp_loadings < 0].sort_values(ascending=True).head(top_n)
    return positive_genes, negative_genes

regions = ['ACC', 'NAc', 'HIP', 'CN', 'mdTN', 'dlPFC', 'EC', 'IPP', 'M1', 'lCb', 'MB']

# ========== PCA: Positive score → positive loadings ==========
print("\n" + "="*70)
print("PCA REGION MARKER GENES")
print("="*70)

for pc in pca_reg_comps:
    pc_row = pca_analysis[pca_analysis['Component'] == pc].iloc[0]
    
    print(f"\n{pc}: {pc_row['Dominant_Region']} (η²={pc_row['Region_eta2']:.2f})")
    
    # Get region scores
    region_scores = {reg: pc_row[f'Reg_{reg}'] for reg in regions}
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1])
    
    pos_genes, neg_genes = extract_marker_genes(pca_loadings, pc, top_n=50)
    
    # PCA RULE: Positive score → positive loadings
    top3_pos = [reg for reg, score in sorted_regions[-3:]]
    top3_neg = [reg for reg, score in sorted_regions[:3]]
    
    print(f"  POSITIVE loadings (genes HIGH in {', '.join(top3_pos)}):")
    print(f"    Top 20 genes: {', '.join(pos_genes.head(20).index.tolist())}")
    
    print(f"  NEGATIVE loadings (genes HIGH in {', '.join(top3_neg)}):")
    print(f"    Top 20 genes: {', '.join(neg_genes.head(20).index.tolist())}")
    
    # Save
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'regions': ', '.join(top3_pos),
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'regions': ', '.join(top3_neg),
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'pca_{pc}_region_markers.csv', index=False)

print(f"\nSaved {len(pca_reg_comps)} PCA region marker files")

# ========== FA: OPPOSITE RULE - Positive score → NEGATIVE loadings ==========
print("\n" + "="*70)
print("FA REGION MARKER GENES")
print("="*70)

for factor in fa_reg_comps:
    fa_row = fa_analysis[fa_analysis['Component'] == factor].iloc[0]
    
    print(f"\n{factor}: {fa_row['Dominant_Region']} (η²={fa_row['Region_eta2']:.2f})")
    
    # Get region scores
    region_scores = {reg: fa_row[f'Reg_{reg}'] for reg in regions}
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1])
    
    pos_genes, neg_genes = extract_marker_genes(fa_loadings, factor, top_n=50)
    
    # FA RULE: OPPOSITE - Positive score → NEGATIVE loadings
    top3_pos = [reg for reg, score in sorted_regions[-3:]]
    top3_neg = [reg for reg, score in sorted_regions[:3]]
    
    print(f"  POSITIVE loadings (genes HIGH in {', '.join(top3_neg)}):")  # SWAPPED
    print(f"    Top 20 genes: {', '.join(pos_genes.head(20).index.tolist())}")
    
    print(f"  NEGATIVE loadings (genes HIGH in {', '.join(top3_pos)}):")  # SWAPPED
    print(f"    Top 20 genes: {', '.join(neg_genes.head(20).index.tolist())}")
    
    # Save
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'regions': ', '.join(top3_neg),  # SWAPPED
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'regions': ', '.join(top3_pos),  # SWAPPED
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'fa_{factor}_region_markers.csv', index=False)

print(f"\nSaved {len(fa_reg_comps)} FA region marker files")

print("\n" + "="*70)
print("Complete!")
print("="*70)

