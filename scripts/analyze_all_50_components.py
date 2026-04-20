#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ANALYZING ALL 50 COMPONENTS - CELL TYPE AND REGION")
print("="*70)

# Load data
pca_df = pd.read_csv('pca_combined_all_celltypes_50pcs.csv')
fa_df = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

pca_sources = pd.read_csv('pc_sources_of_variation_50.csv')
fa_sources = pd.read_csv('fa_sources_of_variation_50.csv')

cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
regions = pca_df['region'].unique()

print(f"\nCell types: {cell_types}")
print(f"Regions: {len(regions)} - {list(regions)}")

# ========== FUNCTION: GET MEANS ==========
def get_celltype_means(df, component_name):
    """Get mean scores for each cell type"""
    means = {}
    for ct in cell_types:
        ct_data = df[df['cell_type'] == ct][component_name]
        means[ct] = ct_data.mean()
    
    abs_means = {ct: abs(val) for ct, val in means.items()}
    dominant_ct = max(abs_means, key=abs_means.get)
    
    return dominant_ct, means

def get_region_means(df, component_name, regions):
    """Get mean scores for each region"""
    means = {}
    for reg in regions:
        reg_data = df[df['region'] == reg][component_name]
        means[reg] = reg_data.mean()
    
    abs_means = {reg: abs(val) for reg, val in means.items()}
    dominant_reg = max(abs_means, key=abs_means.get)
    
    # Get top 3
    sorted_means = sorted(abs_means.items(), key=lambda x: x[1], reverse=True)
    top3_regions = [reg for reg, val in sorted_means[:3]]
    
    return dominant_reg, means, top3_regions

# ========== ANALYZE ALL 50 PCs ==========
print("\n" + "="*70)
print("ANALYZING ALL 50 PCs")
print("="*70)

pca_results = []

for i in range(50):
    pc_name = f'PC{i+1}'
    print(f"Processing {pc_name}...", end=' ')
    
    # Get cell type info
    dominant_ct, ct_means = get_celltype_means(pca_df, pc_name)
    
    # Get region info
    dominant_reg, reg_means, top3_reg = get_region_means(pca_df, pc_name, regions)
    
    # Get metadata
    pc_row = pca_sources[pca_sources['PC'] == pc_name].iloc[0]
    
    result = {
        'Component': pc_name,
        'CellType_eta2': pc_row['celltype_eta2'],
        'Region_eta2': pc_row['region_eta2'],
        'Age_r': pc_row['age_r'],
        'Age_p': pc_row['age_p'],
        'Dominant_CellType': dominant_ct,
        'Dominant_Region': dominant_reg,
        'Top3_Regions': ', '.join(top3_reg),
        **{f'CT_{ct}': val for ct, val in ct_means.items()},
        **{f'Reg_{reg}': val for reg, val in reg_means.items()}
    }
    
    pca_results.append(result)
    print("Done")

pca_all_df = pd.DataFrame(pca_results)
pca_all_df.to_csv('pca_all_50_components_analyzed.csv', index=False)
print("\nSaved: pca_all_50_components_analyzed.csv")

# ========== ANALYZE ALL 50 FACTORS ==========
print("\n" + "="*70)
print("ANALYZING ALL 50 FACTORS")
print("="*70)

fa_results = []

for i in range(50):
    factor_name = f'Factor{i+1}'
    print(f"Processing {factor_name}...", end=' ')
    
    # Get cell type info
    dominant_ct, ct_means = get_celltype_means(fa_df, factor_name)
    
    # Get region info
    dominant_reg, reg_means, top3_reg = get_region_means(fa_df, factor_name, regions)
    
    # Get metadata
    fa_row = fa_sources[fa_sources['Factor'] == factor_name].iloc[0]
    
    result = {
        'Component': factor_name,
        'CellType_eta2': fa_row['celltype_eta2'],
        'Region_eta2': fa_row['region_eta2'],
        'Age_r': fa_row['age_r'],
        'Age_p': fa_row['age_p'],
        'Dominant_CellType': dominant_ct,
        'Dominant_Region': dominant_reg,
        'Top3_Regions': ', '.join(top3_reg),
        **{f'CT_{ct}': val for ct, val in ct_means.items()},
        **{f'Reg_{reg}': val for reg, val in reg_means.items()}
    }
    
    fa_results.append(result)
    print("Done")

fa_all_df = pd.DataFrame(fa_results)
fa_all_df.to_csv('fa_all_50_factors_analyzed.csv', index=False)
print("\nSaved: fa_all_50_factors_analyzed.csv")

# ========== CREATE COMPREHENSIVE HEATMAPS ==========
print("\n" + "="*70)
print("Creating comprehensive heatmaps...")
print("="*70)

# Cell Type Heatmap - ALL 50
fig, axes = plt.subplots(2, 1, figsize=(24, 10))

# PCA cell types
ax = axes[0]
ct_cols = [f'CT_{ct}' for ct in cell_types]
heatmap_data = pca_all_df[ct_cols].T
heatmap_data.index = cell_types
heatmap_data.columns = [f'PC{i+1}' for i in range(50)]

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Mean Score'},
            ax=ax, xticklabels=True, yticklabels=True)
ax.set_xlabel('PC', fontweight='bold', fontsize=12)
ax.set_ylabel('Cell Type', fontweight='bold', fontsize=12)
ax.set_title('PCA: Cell Type Scores Across All 50 Components', 
             fontweight='bold', fontsize=13)
plt.setp(ax.get_xticklabels(), fontsize=7)

# FA cell types
ax = axes[1]
heatmap_data = fa_all_df[ct_cols].T
heatmap_data.index = cell_types
heatmap_data.columns = [f'F{i+1}' for i in range(50)]

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Mean Score'},
            ax=ax, xticklabels=True, yticklabels=True)
ax.set_xlabel('Factor', fontweight='bold', fontsize=12)
ax.set_ylabel('Cell Type', fontweight='bold', fontsize=12)
ax.set_title('FA: Cell Type Scores Across All 50 Factors', 
             fontweight='bold', fontsize=13)
plt.setp(ax.get_xticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig('all_50_celltype_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: all_50_celltype_heatmap.png")

# Region Heatmap - ALL 50
fig, axes = plt.subplots(2, 1, figsize=(24, 14))

# PCA regions
ax = axes[0]
reg_cols = [f'Reg_{reg}' for reg in regions]
heatmap_data = pca_all_df[reg_cols].T
heatmap_data.index = regions
heatmap_data.columns = [f'PC{i+1}' for i in range(50)]

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Mean Score'},
            ax=ax, xticklabels=True, yticklabels=True)
ax.set_xlabel('PC', fontweight='bold', fontsize=12)
ax.set_ylabel('Region', fontweight='bold', fontsize=12)
ax.set_title('PCA: Region Scores Across All 50 Components', 
             fontweight='bold', fontsize=13)
plt.setp(ax.get_xticklabels(), fontsize=7)

# FA regions
ax = axes[1]
heatmap_data = fa_all_df[reg_cols].T
heatmap_data.index = regions
heatmap_data.columns = [f'F{i+1}' for i in range(50)]

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Mean Score'},
            ax=ax, xticklabels=True, yticklabels=True)
ax.set_xlabel('Factor', fontweight='bold', fontsize=12)
ax.set_ylabel('Region', fontweight='bold', fontsize=12)
ax.set_title('FA: Region Scores Across All 50 Factors', 
             fontweight='bold', fontsize=13)
plt.setp(ax.get_xticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig('all_50_region_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: all_50_region_heatmap.png")

# ========== MERGE WITH GO TERMS ==========
print("\n" + "="*70)
print("Creating complete interpretation files...")
print("="*70)

# Load GO enrichment
pca_go = pd.read_csv('pca_interpretation_summary.csv')
fa_go = pd.read_csv('fa_interpretation_summary.csv')

# Merge everything
pca_complete = pca_go.merge(
    pca_all_df[['Component', 'Dominant_CellType', 'Dominant_Region', 'Top3_Regions']], 
    on='Component', how='left'
)
pca_complete.to_csv('pca_complete_interpretation_all_50.csv', index=False)
print("Saved: pca_complete_interpretation_all_50.csv")

fa_complete = fa_go.merge(
    fa_all_df[['Component', 'Dominant_CellType', 'Dominant_Region', 'Top3_Regions']], 
    on='Component', how='left'
)
fa_complete.to_csv('fa_complete_interpretation_all_50.csv', index=False)
print("Saved: fa_complete_interpretation_all_50.csv")

# ========== SUMMARY STATISTICS ==========
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nPCA - Components by dominant source:")
print(f"  Cell type dominated (η² > 0.3): {(pca_all_df['CellType_eta2'] > 0.3).sum()}")
print(f"  Region dominated (η² > 0.3): {(pca_all_df['Region_eta2'] > 0.3).sum()}")
print(f"  Age significant (p < 0.05): {(pca_all_df['Age_p'] < 0.05).sum()}")

print("\nFA - Factors by dominant source:")
print(f"  Cell type dominated (η² > 0.3): {(fa_all_df['CellType_eta2'] > 0.3).sum()}")
print(f"  Region dominated (η² > 0.3): {(fa_all_df['Region_eta2'] > 0.3).sum()}")
print(f"  Age significant (p < 0.05): {(fa_all_df['Age_p'] < 0.05).sum()}")

print("\nPCA - Dominant cell type distribution:")
print(pca_all_df['Dominant_CellType'].value_counts())

print("\nFA - Dominant cell type distribution:")
print(fa_all_df['Dominant_CellType'].value_counts())

print("\nPCA - Dominant region distribution:")
print(pca_all_df['Dominant_Region'].value_counts())

print("\nFA - Dominant region distribution:")
print(fa_all_df['Dominant_Region'].value_counts())

print("\n" + "="*70)
print("Complete!")
print("="*70)

