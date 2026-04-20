#!/usr/bin/env python

import pandas as pd
import numpy as np

print("="*70)
print("EXTRACTING CELL TYPE MARKER GENES - SEPARATE LOGIC FOR PCA VS FA")
print("="*70)

# Load loadings
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)

# Load component analysis
pca_analysis = pd.read_csv('pca_all_50_components_analyzed.csv')
fa_analysis = pd.read_csv('fa_all_50_factors_analyzed.csv')

# Get cell type dominated components
pca_ct_comps = pca_analysis[pca_analysis['CellType_eta2'] > 0.3]['Component'].tolist()
fa_ct_comps = fa_analysis[fa_analysis['CellType_eta2'] > 0.3]['Component'].tolist()

print(f"\nAnalyzing {len(pca_ct_comps)} PCA components and {len(fa_ct_comps)} FA factors")

def extract_marker_genes(loadings_df, component, top_n=50):
    comp_loadings = loadings_df.loc[component]
    positive_genes = comp_loadings[comp_loadings > 0].sort_values(ascending=False).head(top_n)
    negative_genes = comp_loadings[comp_loadings < 0].sort_values(ascending=True).head(top_n)
    return positive_genes, negative_genes

# ========== PCA: Positive score → Positive loadings ==========
print("\n" + "="*70)
print("PCA CELL TYPE MARKER GENES")
print("="*70)

for pc in pca_ct_comps:
    pc_row = pca_analysis[pca_analysis['Component'] == pc].iloc[0]
    
    print(f"\n{pc}: {pc_row['Dominant_CellType']} (η²={pc_row['CellType_eta2']:.2f})")
    
    ct_scores = {
        'Glutamatergic': pc_row['CT_Glutamatergic'],
        'GABAergic': pc_row['CT_GABAergic'],
        'Astrocytes': pc_row['CT_Astrocytes'],
        'Microglia': pc_row['CT_Microglia']
    }
    
    for ct, score in ct_scores.items():
        print(f"    {ct}: {score:.1f}")
    
    pos_genes, neg_genes = extract_marker_genes(pca_loadings, pc, top_n=50)
    
    # PCA RULE: Positive score → positive loadings
    pos_score_cts = [ct for ct, score in ct_scores.items() if score > 0]
    neg_score_cts = [ct for ct, score in ct_scores.items() if score < 0]
    
    print(f"\n  POSITIVE loadings (genes HIGH in {', '.join(pos_score_cts)}):")
    print(f"    Top 20 genes: {', '.join(pos_genes.head(20).index.tolist())}")
    
    print(f"\n  NEGATIVE loadings (genes HIGH in {', '.join(neg_score_cts)}):")
    print(f"    Top 20 genes: {', '.join(neg_genes.head(20).index.tolist())}")
    
    # Save
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'cell_type': ', '.join(pos_score_cts),
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'cell_type': ', '.join(neg_score_cts),
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'pca_{pc}_celltype_markers.csv', index=False)
    print(f"  Saved: pca_{pc}_celltype_markers.csv")

# ========== FA: OPPOSITE RULE - Positive score → NEGATIVE loadings ==========
print("\n" + "="*70)
print("FA CELL TYPE MARKER GENES")
print("="*70)

for factor in fa_ct_comps:
    fa_row = fa_analysis[fa_analysis['Component'] == factor].iloc[0]
    
    print(f"\n{factor}: {fa_row['Dominant_CellType']} (η²={fa_row['CellType_eta2']:.2f})")
    
    ct_scores = {
        'Glutamatergic': fa_row['CT_Glutamatergic'],
        'GABAergic': fa_row['CT_GABAergic'],
        'Astrocytes': fa_row['CT_Astrocytes'],
        'Microglia': fa_row['CT_Microglia']
    }
    
    for ct, score in ct_scores.items():
        print(f"    {ct}: {score:.1f}")
    
    pos_genes, neg_genes = extract_marker_genes(fa_loadings, factor, top_n=50)
    
    # FA RULE: OPPOSITE - Positive score → NEGATIVE loadings
    pos_score_cts = [ct for ct, score in ct_scores.items() if score > 0]
    neg_score_cts = [ct for ct, score in ct_scores.items() if score < 0]
    
    print(f"\n  POSITIVE loadings (genes HIGH in {', '.join(neg_score_cts)}):")
    print(f"    Top 20 genes: {', '.join(pos_genes.head(20).index.tolist())}")
    
    print(f"\n  NEGATIVE loadings (genes HIGH in {', '.join(pos_score_cts)}):")
    print(f"    Top 20 genes: {', '.join(neg_genes.head(20).index.tolist())}")
    
    # Save
    pos_df = pd.DataFrame({
        'gene': pos_genes.index,
        'loading': pos_genes.values,
        'cell_type': ', '.join(neg_score_cts),  # SWAPPED
        'direction': 'positive'
    })
    
    neg_df = pd.DataFrame({
        'gene': neg_genes.index,
        'loading': neg_genes.values,
        'cell_type': ', '.join(pos_score_cts),  # SWAPPED
        'direction': 'negative'
    })
    
    combined = pd.concat([pos_df, neg_df])
    combined.to_csv(f'fa_{factor}_celltype_markers.csv', index=False)
    print(f"  Saved: fa_{factor}_celltype_markers.csv")

print("\n" + "="*70)
print("Complete!")
print("="*70)

