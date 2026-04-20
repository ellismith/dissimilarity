#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("IDENTIFYING WHICH REGION EACH COMPONENT REPRESENTS")
print("="*70)

# Load data
pca_df = pd.read_csv('pca_combined_all_celltypes_50pcs.csv')
fa_df = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

pca_sources = pd.read_csv('pc_sources_of_variation_50.csv')
fa_sources = pd.read_csv('fa_sources_of_variation_50.csv')

# Get region dominated components (η² > 0.3)
pca_reg_components = pca_sources[pca_sources['region_eta2'] > 0.3]['PC'].tolist()
fa_reg_components = fa_sources[fa_sources['region_eta2'] > 0.3]['Factor'].tolist()

print(f"\nPCA: {len(pca_reg_components)} region-dominated components (η² > 0.3)")
print(f"FA: {len(fa_reg_components)} region-dominated components (η² > 0.3)")

# Get list of all regions
regions = pca_df['region'].unique()
print(f"\nRegions in dataset: {len(regions)}")
print(regions)

# ========== FUNCTION: IDENTIFY REGION ==========
def identify_region_for_component(df, component_name, regions):
    """Determine which region has highest mean score for this component"""
    
    means = {}
    
    for reg in regions:
        reg_data = df[df['region'] == reg][component_name]
        means[reg] = reg_data.mean()
    
    # Find region with highest absolute mean
    abs_means = {reg: abs(val) for reg, val in means.items()}
    dominant_reg = max(abs_means, key=abs_means.get)
    
    return dominant_reg, means

# ========== ANALYZE PCA ==========
print("\n" + "="*70)
print("PCA REGION IDENTIFICATION")
print("="*70)

pca_identifications = []

for pc in pca_reg_components:
    dominant_reg, means = identify_region_for_component(pca_df, pc, regions)
    
    # Get metadata
    pc_row = pca_sources[pca_sources['PC'] == pc].iloc[0]
    
    # Get top 3 regions
    sorted_means = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)
    top3_regions = [reg for reg, val in sorted_means[:3]]
    
    pca_identifications.append({
        'Component': pc,
        'Dominant_Region': dominant_reg,
        'Top3_Regions': ', '.join(top3_regions),
        'Region_eta2': pc_row['region_eta2'],
        'CellType_eta2': pc_row['celltype_eta2'],
        **{f'{reg}_mean': val for reg, val in means.items()}
    })
    
    print(f"\n{pc}: {dominant_reg}")
    print(f"  Region η²: {pc_row['region_eta2']:.2f}, Cell type η²: {pc_row['celltype_eta2']:.2f}")
    print(f"  Top 3 regions by absolute mean:")
    for i, (reg, val) in enumerate(sorted_means[:3], 1):
        print(f"    {i}. {reg}: {val:.3f}")

pca_id_df = pd.DataFrame(pca_identifications)
pca_id_df.to_csv('pca_region_identification.csv', index=False)
print("\nSaved: pca_region_identification.csv")

# ========== ANALYZE FA ==========
print("\n" + "="*70)
print("FA REGION IDENTIFICATION")
print("="*70)

fa_identifications = []

for factor in fa_reg_components:
    dominant_reg, means = identify_region_for_component(fa_df, factor, regions)
    
    # Get metadata
    fa_row = fa_sources[fa_sources['Factor'] == factor].iloc[0]
    
    # Get top 3 regions
    sorted_means = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)
    top3_regions = [reg for reg, val in sorted_means[:3]]
    
    fa_identifications.append({
        'Component': factor,
        'Dominant_Region': dominant_reg,
        'Top3_Regions': ', '.join(top3_regions),
        'Region_eta2': fa_row['region_eta2'],
        'CellType_eta2': fa_row['celltype_eta2'],
        **{f'{reg}_mean': val for reg, val in means.items()}
    })
    
    print(f"\n{factor}: {dominant_reg}")
    print(f"  Region η²: {fa_row['region_eta2']:.2f}, Cell type η²: {fa_row['celltype_eta2']:.2f}")
    print(f"  Top 3 regions by absolute mean:")
    for i, (reg, val) in enumerate(sorted_means[:3], 1):
        print(f"    {i}. {reg}: {val:.3f}")

fa_id_df = pd.DataFrame(fa_identifications)
fa_id_df.to_csv('fa_region_identification.csv', index=False)
print("\nSaved: fa_region_identification.csv")

# ========== CREATE HEATMAP ==========
print("\n" + "="*70)
print("Creating region heatmap...")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# PCA heatmap
ax = axes[0]
if len(pca_identifications) > 0:
    # Create matrix: components × regions
    region_cols = [col for col in pca_id_df.columns if col.endswith('_mean')]
    region_names = [col.replace('_mean', '') for col in region_cols]
    
    heatmap_data = pca_id_df[region_cols].T
    heatmap_data.index = region_names
    heatmap_data.columns = pca_id_df['Component']
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Mean Score'},
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('PC', fontweight='bold')
    ax.set_ylabel('Region', fontweight='bold')
    ax.set_title('PCA: Region Scores for Region-Dominated Components (η² > 0.3)', 
                 fontweight='bold', fontsize=12)

# FA heatmap
ax = axes[1]
if len(fa_identifications) > 0:
    region_cols = [col for col in fa_id_df.columns if col.endswith('_mean')]
    region_names = [col.replace('_mean', '') for col in region_cols]
    
    heatmap_data = fa_id_df[region_cols].T
    heatmap_data.index = region_names
    heatmap_data.columns = fa_id_df['Component']
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Score'},
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Factor', fontweight='bold')
    ax.set_ylabel('Region', fontweight='bold')
    ax.set_title('FA: Region Scores for Region-Dominated Factors (η² > 0.3)', 
                 fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('region_component_identification.png', dpi=300, bbox_inches='tight')
print("Saved: region_component_identification.png")

# ========== MERGE WITH GO TERMS ==========
print("\n" + "="*70)
print("Merging with GO enrichment...")
print("="*70)

# Load previous summaries
pca_complete = pd.read_csv('pca_complete_interpretation.csv')
fa_complete = pd.read_csv('fa_complete_interpretation.csv')

# Merge region info
pca_complete = pca_complete.merge(
    pca_id_df[['Component', 'Dominant_Region', 'Top3_Regions']], 
    on='Component', how='left'
)
pca_complete.to_csv('pca_complete_interpretation.csv', index=False)

fa_complete = fa_complete.merge(
    fa_id_df[['Component', 'Dominant_Region', 'Top3_Regions']], 
    on='Component', how='left'
)
fa_complete.to_csv('fa_complete_interpretation.csv', index=False)

print("Updated: pca_complete_interpretation.csv")
print("Updated: fa_complete_interpretation.csv")

# Print summary
print("\n" + "="*70)
print("REGION-DOMINATED COMPONENTS SUMMARY")
print("="*70)

print("\nPCA Components:")
for idx, row in pca_id_df.iterrows():
    # Get GO term
    go_row = pca_complete[pca_complete['Component'] == row['Component']].iloc[0]
    print(f"{row['Component']}: {row['Dominant_Region']} - {go_row['Top_GO_term']}")
    print(f"  Top 3: {row['Top3_Regions']}")

print("\nFA Factors:")
for idx, row in fa_id_df.iterrows():
    # Get GO term
    go_row = fa_complete[fa_complete['Component'] == row['Component']].iloc[0]
    print(f"{row['Component']}: {row['Dominant_Region']} - {go_row['Top_GO_term']}")
    print(f"  Top 3: {row['Top3_Regions']}")

print("\n" + "="*70)
print("Complete!")
print("="*70)

