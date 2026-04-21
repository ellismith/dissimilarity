#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("IDENTIFYING WHICH CELL TYPE EACH COMPONENT REPRESENTS")
print("="*70)

# Load data
pca_df = pd.read_csv('pca_combined_all_celltypes_50pcs.csv')
fa_df = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

pca_sources = pd.read_csv('pc_sources_of_variation_50.csv')
fa_sources = pd.read_csv('fa_sources_of_variation_50.csv')

# Get cell type dominated components (η² > 0.3)
pca_ct_components = pca_sources[pca_sources['celltype_eta2'] > 0.3]['PC'].tolist()
fa_ct_components = fa_sources[fa_sources['celltype_eta2'] > 0.3]['Factor'].tolist()

print(f"\nPCA: {len(pca_ct_components)} cell-type dominated components (η² > 0.3)")
print(f"FA: {len(fa_ct_components)} cell-type dominated components (η² > 0.3)")

# ========== FUNCTION: IDENTIFY CELL TYPE ==========
def identify_celltype_for_component(df, component_name):
    """Determine which cell type has highest mean score for this component"""
    
    cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
    means = {}
    
    for ct in cell_types:
        ct_data = df[df['cell_type'] == ct][component_name]
        means[ct] = ct_data.mean()
    
    # Find cell type with highest absolute mean
    abs_means = {ct: abs(val) for ct, val in means.items()}
    dominant_ct = max(abs_means, key=abs_means.get)
    
    return dominant_ct, means

# ========== ANALYZE PCA ==========
print("\n" + "="*70)
print("PCA CELL TYPE IDENTIFICATION")
print("="*70)

pca_identifications = []

for pc in pca_ct_components:
    dominant_ct, means = identify_celltype_for_component(pca_df, pc)
    
    # Get metadata
    pc_row = pca_sources[pca_sources['PC'] == pc].iloc[0]
    
    pca_identifications.append({
        'Component': pc,
        'CellType': dominant_ct,
        'CellType_eta2': pc_row['celltype_eta2'],
        'Region_eta2': pc_row['region_eta2'],
        'Glutamatergic_mean': means['Glutamatergic'],
        'GABAergic_mean': means['GABAergic'],
        'Astrocytes_mean': means['Astrocytes'],
        'Microglia_mean': means['Microglia']
    })
    
    print(f"\n{pc}: {dominant_ct}")
    print(f"  Cell type η²: {pc_row['celltype_eta2']:.2f}, Region η²: {pc_row['region_eta2']:.2f}")
    print(f"  Means by cell type:")
    for ct, val in means.items():
        marker = " <-- HIGHEST" if ct == dominant_ct else ""
        print(f"    {ct}: {val:.3f}{marker}")

pca_id_df = pd.DataFrame(pca_identifications)
pca_id_df.to_csv('pca_celltype_identification.csv', index=False)
print("\nSaved: pca_celltype_identification.csv")

# ========== ANALYZE FA ==========
print("\n" + "="*70)
print("FA CELL TYPE IDENTIFICATION")
print("="*70)

fa_identifications = []

for factor in fa_ct_components:
    dominant_ct, means = identify_celltype_for_component(fa_df, factor)
    
    # Get metadata
    fa_row = fa_sources[fa_sources['Factor'] == factor].iloc[0]
    
    fa_identifications.append({
        'Component': factor,
        'CellType': dominant_ct,
        'CellType_eta2': fa_row['celltype_eta2'],
        'Region_eta2': fa_row['region_eta2'],
        'Glutamatergic_mean': means['Glutamatergic'],
        'GABAergic_mean': means['GABAergic'],
        'Astrocytes_mean': means['Astrocytes'],
        'Microglia_mean': means['Microglia']
    })
    
    print(f"\n{factor}: {dominant_ct}")
    print(f"  Cell type η²: {fa_row['celltype_eta2']:.2f}, Region η²: {fa_row['region_eta2']:.2f}")
    print(f"  Means by cell type:")
    for ct, val in means.items():
        marker = " <-- HIGHEST" if ct == dominant_ct else ""
        print(f"    {ct}: {val:.3f}{marker}")

fa_id_df = pd.DataFrame(fa_identifications)
fa_id_df.to_csv('fa_celltype_identification.csv', index=False)
print("\nSaved: fa_celltype_identification.csv")

# ========== CREATE HEATMAP ==========
print("\n" + "="*70)
print("Creating cell type heatmap...")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# PCA heatmap
ax = axes[0]
if len(pca_identifications) > 0:
    # Create matrix: cell types × components
    celltype_cols = ['Glutamatergic_mean', 'GABAergic_mean', 'Astrocytes_mean', 'Microglia_mean']
    celltype_names = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
    
    heatmap_data = pca_id_df[celltype_cols].T
    heatmap_data.index = celltype_names
    heatmap_data.columns = pca_id_df['Component']
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Mean Score'},
                ax=ax, linewidths=0.5, linecolor='gray',
                annot=False)
    ax.set_xlabel('PC', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cell Type', fontweight='bold', fontsize=11)
    ax.set_title('PCA: Cell Type Scores for Cell-Type Dominated Components (η² > 0.3)', 
                 fontweight='bold', fontsize=12)

# FA heatmap
ax = axes[1]
if len(fa_identifications) > 0:
    celltype_cols = ['Glutamatergic_mean', 'GABAergic_mean', 'Astrocytes_mean', 'Microglia_mean']
    celltype_names = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
    
    heatmap_data = fa_id_df[celltype_cols].T
    heatmap_data.index = celltype_names
    heatmap_data.columns = fa_id_df['Component']
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Score'},
                ax=ax, linewidths=0.5, linecolor='gray',
                annot=False)
    ax.set_xlabel('Factor', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cell Type', fontweight='bold', fontsize=11)
    ax.set_title('FA: Cell Type Scores for Cell-Type Dominated Factors (η² > 0.3)', 
                 fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('celltype_component_identification_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: celltype_component_identification_heatmap.png")

# ========== MERGE WITH GO TERMS ==========
print("\n" + "="*70)
print("Merging with GO enrichment...")
print("="*70)

# Load or create complete interpretation
try:
    pca_complete = pd.read_csv('pca_complete_interpretation.csv')
except:
    pca_complete = pd.read_csv('pca_interpretation_summary.csv')

try:
    fa_complete = pd.read_csv('fa_complete_interpretation.csv')
except:
    fa_complete = pd.read_csv('fa_interpretation_summary.csv')

# Update cell type info
pca_complete = pca_complete.drop(columns=['CellType'], errors='ignore')
pca_complete = pca_complete.merge(pca_id_df[['Component', 'CellType']], 
                                  on='Component', how='left')
pca_complete.to_csv('pca_complete_interpretation.csv', index=False)

fa_complete = fa_complete.drop(columns=['CellType'], errors='ignore')
fa_complete = fa_complete.merge(fa_id_df[['Component', 'CellType']], 
                                on='Component', how='left')
fa_complete.to_csv('fa_complete_interpretation.csv', index=False)

print("Updated: pca_complete_interpretation.csv")
print("Updated: fa_complete_interpretation.csv")

# Print summary
print("\n" + "="*70)
print("COMPLETE CELL TYPE INTERPRETATIONS")
print("="*70)

print("\nPCA Components:")
for idx, row in pca_complete[pca_complete['CellType'].notna()].iterrows():
    print(f"{row['Component']}: {row['CellType']} (η²={row['CellType_eta2']:.2f}) - {row['Top_GO_term']}")

print("\nFA Factors:")
for idx, row in fa_complete[fa_complete['CellType'].notna()].iterrows():
    print(f"{row['Component']}: {row['CellType']} (η²={row['CellType_eta2']:.2f}) - {row['Top_GO_term']}")

print("\n" + "="*70)
print("Complete!")
print("="*70)

