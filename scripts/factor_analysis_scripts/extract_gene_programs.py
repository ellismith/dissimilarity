#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import scanpy as sc
from scipy import stats

def get_top_genes_from_factor(loadings, gene_names, factor_idx, n_top=100):
    """Extract top genes (both positive and negative) for a given factor"""
    factor_loadings = loadings[:, factor_idx]
    
    # Get indices sorted by absolute loading
    abs_sorted_idx = np.argsort(np.abs(factor_loadings))[::-1]
    
    # Get top positive and negative separately
    pos_idx = np.argsort(factor_loadings)[::-1][:n_top]
    neg_idx = np.argsort(factor_loadings)[:n_top]
    
    top_genes_df = pd.DataFrame({
        'gene': gene_names[abs_sorted_idx[:n_top]],
        'loading': factor_loadings[abs_sorted_idx[:n_top]],
        'abs_loading': np.abs(factor_loadings[abs_sorted_idx[:n_top]])
    })
    
    return top_genes_df

def analyze_region_factor(cell_type, region, factor_num, adata_path):
    """Analyze gene loadings for a specific region and factor"""
    
    print(f"\n{'='*70}")
    print(f"Analyzing {cell_type} - {region} - Factor{factor_num}")
    print(f"{'='*70}")
    
    # Load data
    adata = sc.read_h5ad(adata_path, backed='r')
    
    # Get metadata
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    obs_df['sex'] = obs_df['sex'].astype(str)
    
    # Filter to this region
    region_mask = obs_df['region'] == region
    region_obs = obs_df[region_mask].copy()
    region_obs['pseudobulk_id'] = region_obs['animal_id']
    
    animal_samples = region_obs[['pseudobulk_id', 'animal_id', 'region', 'age', 'sex']].drop_duplicates()
    
    # Create pseudobulk
    pseudobulk_dict = {}
    for animal_id in animal_samples['animal_id']:
        animal_mask = region_obs['animal_id'] == animal_id
        cell_mask_full = region_mask.copy()
        cell_mask_full[region_mask] = animal_mask.values
        cell_indices = np.where(cell_mask_full)[0]
        pseudobulk_dict[animal_id] = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
    
    pseudobulk_matrix = np.vstack([pseudobulk_dict[aid] for aid in animal_samples['animal_id']])
    
    # Filter genes
    min_animals = int(0.2 * len(animal_samples))
    gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
    genes_keep = gene_counts >= min_animals
    pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]
    gene_names_filtered = adata.var_names[genes_keep]
    
    # Log-normalize
    cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
    pseudobulk_log = np.log2(cpm + 1)
    
    # Standardize
    scaler = StandardScaler()
    pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)
    
    # Run Factor Analysis
    n_factors = 10
    fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
    factors = fa.fit_transform(pseudobulk_scaled)
    
    # Get loadings (components_)
    loadings = fa.components_.T  # Shape: (n_genes, n_factors)
    
    # Extract top genes for the specified factor
    factor_idx = factor_num - 1  # Convert to 0-indexed
    top_genes = get_top_genes_from_factor(loadings, gene_names_filtered, factor_idx, n_top=100)
    
    # Correlate this factor with age
    fa_df = pd.DataFrame(factors, columns=[f'Factor{i+1}' for i in range(n_factors)])
    fa_df['age'] = animal_samples['age'].values
    age_corr, age_pval = stats.pearsonr(fa_df[f'Factor{factor_num}'], fa_df['age'])
    
    print(f"\nFactor{factor_num} correlation with age: r = {age_corr:.3f}, p = {age_pval:.3e}")
    print(f"\nTop 20 genes (by absolute loading):")
    print(top_genes.head(20).to_string(index=False))
    
    # Save full gene list
    output_file = f'gene_loadings_{cell_type}_{region}_Factor{factor_num}.csv'
    top_genes.to_csv(output_file, index=False)
    print(f"\nSaved full gene list to: {output_file}")
    
    return top_genes, age_corr

# Analyze top factors from each cell type and region
print("="*70)
print("EXTRACTING GENE PROGRAMS FROM TOP AGE-ASSOCIATED FACTORS")
print("="*70)

# Top hits to analyze:
analyses = [
    # Cell type, region, factor number, adata path
    ('GABAergic', 'dlPFC', 10, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'),
    ('GABAergic', 'ACC', 7, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'),
    ('GABAergic', 'M1', 3, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'),
    ('Astrocytes', 'CN', 7, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad'),
    ('Astrocytes', 'M1', 8, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad'),
    ('Astrocytes', 'ACC', 6, '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad'),
]

all_results = []

for cell_type, region, factor_num, adata_path in analyses:
    try:
        top_genes, age_corr = analyze_region_factor(cell_type, region, factor_num, adata_path)
        all_results.append({
            'cell_type': cell_type,
            'region': region,
            'factor': f'Factor{factor_num}',
            'age_correlation': age_corr,
            'top_genes': ', '.join(top_genes.head(10)['gene'].tolist())
        })
    except Exception as e:
        print(f"\nError analyzing {cell_type} {region} Factor{factor_num}: {e}")

# Create summary
print("\n" + "="*70)
print("SUMMARY OF TOP AGE-ASSOCIATED GENE PROGRAMS")
print("="*70)
summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))
summary_df.to_csv('age_associated_gene_programs_summary.csv', index=False)
print("\nSaved summary to: age_associated_gene_programs_summary.csv")

print("\n" + "="*70)
print("Next steps:")
print("1. Run pathway enrichment (GO/KEGG) on these gene lists")
print("2. Look for common genes across regions/cell types")
print("3. Validate specific genes with known aging markers")
print("="*70)

