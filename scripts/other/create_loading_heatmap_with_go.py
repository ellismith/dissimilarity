#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CREATING GENE LOADING HEATMAPS")
print("="*70)

# ========== LOAD DATA ==========
print("\nLoading PCA loadings with gene symbols...")
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)

print("\nLoading FA loadings with gene symbols...")
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)

# Load sources of variation
pca_sources = pd.read_csv('pc_sources_of_variation_50.csv')
fa_sources = pd.read_csv('fa_sources_of_variation_50.csv')

# ========== FUNCTION: CREATE COMPONENT HEATMAPS ==========
def create_component_heatmaps(loadings_df, sources_df, method_name, components_to_show=10, top_n=30):
    """Show top genes contributing to each component"""
    
    print(f"\nCreating {method_name} gene contribution plots...")
    
    # Create grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()
    
    for idx in range(components_to_show):
        component = loadings_df.index[idx]
        ax = axes[idx]
        
        # Get top genes by ABSOLUTE loading (strength of contribution)
        comp_loadings = loadings_df.iloc[idx]
        top_genes = comp_loadings.abs().sort_values(ascending=False).head(top_n)
        
        # Plot as horizontal bars using actual values (keep sign for visualization)
        y_pos = np.arange(len(top_genes))
        values = comp_loadings[top_genes.index]
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in values]
        
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes.index, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Loading', fontsize=8)
        
        # Get what the component captures
        row = sources_df[sources_df.iloc[:, 0] == component].iloc[0]
        ct_eta = row['celltype_eta2']
        reg_eta = row['region_eta2']
        
        if ct_eta > 0.5:
            captures = "Cell Type"
        elif reg_eta > 0.3:
            captures = "Region"
        else:
            captures = "Mixed"
        
        ax.set_title(f'{component}: {captures}\nCT η²={ct_eta:.2f}, Reg η²={reg_eta:.2f}',
                    fontsize=8, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'{method_name}: Top Contributing Genes per Component',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'{method_name.lower()}_top_genes_per_component.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ========== FUNCTION: CREATE SIMPLE HEATMAP ==========
def create_simple_heatmap(loadings_df, sources_df, method_name, n_components=15):
    """Simple heatmap showing top genes for each component"""
    
    print(f"\nCreating {method_name} heatmap...")
    
    # Get top 20 genes per component
    top_genes_per_comp = {}
    for i in range(n_components):
        comp = loadings_df.index[i]
        top = loadings_df.iloc[i].abs().sort_values(ascending=False).head(20)
        top_genes_per_comp[comp] = top.index.tolist()
    
    # Get union of all top genes
    all_genes = list(set([g for genes in top_genes_per_comp.values() for g in genes]))
    print(f"Total unique genes across {n_components} components: {len(all_genes)}")
    
    # Create matrix
    components = list(top_genes_per_comp.keys())
    heatmap_data = loadings_df.loc[components, all_genes]
    
    # Create figure
    fig, (ax_heat, ax_anno) = plt.subplots(1, 2, figsize=(20, 10),
                                            gridspec_kw={'width_ratios': [5, 1]})
    
    # Heatmap
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r', 
                center=0,
                vmin=-0.15, vmax=0.15,
                cbar_kws={'label': 'Loading'},
                xticklabels=True,
                yticklabels=True,
                ax=ax_heat)
    
    ax_heat.set_xlabel('Gene', fontweight='bold', fontsize=12)
    ax_heat.set_ylabel('Component', fontweight='bold', fontsize=12)
    ax_heat.set_title(f'{method_name}: Gene Loadings', fontweight='bold', fontsize=13)
    plt.setp(ax_heat.get_xticklabels(), rotation=90, fontsize=6)
    
    # Annotations
    ax_anno.axis('off')
    anno_text = "What each captures:\n" + "="*25 + "\n\n"
    
    for comp in components:
        row = sources_df[sources_df.iloc[:, 0] == comp].iloc[0]
        ct = row['celltype_eta2']
        reg = row['region_eta2']
        
        if ct > 0.5:
            label = "Cell Type"
        elif reg > 0.3:
            label = "Region"
        else:
            label = "Mixed"
        
        anno_text += f"{comp}: {label}\n"
    
    ax_anno.text(0.05, 0.98, anno_text,
                transform=ax_anno.transAxes,
                fontsize=7,
                verticalalignment='top',
                fontfamily='monospace')
    
    plt.tight_layout()
    filename = f'{method_name.lower()}_gene_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ========== CREATE ALL PLOTS ==========
create_component_heatmaps(pca_loadings, pca_sources, 'PCA')
create_component_heatmaps(fa_loadings, fa_sources, 'FA')

create_simple_heatmap(pca_loadings, pca_sources, 'PCA')
create_simple_heatmap(fa_loadings, fa_sources, 'FA')

print("\n" + "="*70)
print("Complete!")
print("="*70)

