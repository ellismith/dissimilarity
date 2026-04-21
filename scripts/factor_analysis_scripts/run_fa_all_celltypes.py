#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scanpy as sc

# Define cell types to analyze
cell_types = {
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad'
}

def analyze_celltype_region(cell_type, adata_path, region):
    """Analyze one cell type in one region"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {cell_type} - {region}")
    print(f"{'='*60}")
    
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
    
    if region_obs.empty:
        print(f"No data for {region}")
        return None
    
    region_obs['pseudobulk_id'] = region_obs['animal_id']
    
    # Get unique animals
    animal_samples = region_obs[['pseudobulk_id', 'animal_id', 'region', 'age', 'sex']].drop_duplicates()
    n_animals = len(animal_samples)
    print(f"Number of animals: {n_animals}")
    
    if n_animals < 20:
        print(f"Skipping - too few animals")
        return None
    
    # Create pseudobulk
    print("Aggregating by animal...")
    pseudobulk_dict = {}
    
    for animal_id in animal_samples['animal_id']:
        animal_mask = region_obs['animal_id'] == animal_id
        cell_mask_full = region_mask.copy()
        cell_mask_full[region_mask] = animal_mask.values
        cell_indices = np.where(cell_mask_full)[0]
        pseudobulk_dict[animal_id] = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
    
    pseudobulk_matrix = np.vstack([pseudobulk_dict[aid] for aid in animal_samples['animal_id']])
    
    # Filter genes
    min_animals = int(0.2 * n_animals)
    gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
    genes_keep = gene_counts >= min_animals
    n_genes = genes_keep.sum()
    print(f"Keeping {n_genes} genes")
    
    if n_genes < 1000:
        print(f"Skipping - too few genes")
        return None
    
    pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]
    gene_names_filtered = adata.var_names[genes_keep]
    
    # Log-normalize
    cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
    pseudobulk_log = np.log2(cpm + 1)
    
    # Standardize
    scaler = StandardScaler()
    pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)
    
    # Run Factor Analysis
    n_factors = min(10, n_animals - 5)
    print(f"Running Factor Analysis with {n_factors} factors...")
    
    fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
    factors = fa.fit_transform(pseudobulk_scaled)
    
    # Create results dataframe
    fa_df = pd.DataFrame(
        factors,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    fa_df = pd.concat([animal_samples.reset_index(drop=True), fa_df], axis=1)
    
    # Correlate with age
    print("\n--- Factor correlations with Age ---")
    best_corr = 0
    best_factor = None
    best_pval = 1
    
    for i in range(n_factors):
        factor_name = f'Factor{i+1}'
        corr, pval = stats.pearsonr(fa_df[factor_name], fa_df['age'])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_factor = factor_name
            best_pval = pval
        if pval < 0.05:
            print(f"{factor_name}: r = {corr:.3f}, p = {pval:.3e} ***")
    
    print(f"\nBest age correlation: {best_factor} (r = {best_corr:.3f}, p = {best_pval:.3e})")
    
    # Save region-specific results
    fa_df.to_csv(f'factor_analysis_{cell_type}_{region}.csv', index=False)
    
    # Get gene loadings for best factor if significant
    if best_pval < 0.05:
        loadings = fa.components_.T
        factor_idx = int(best_factor.replace('Factor', '')) - 1
        factor_loadings = loadings[:, factor_idx]
        
        # Sort by absolute loading
        abs_sorted_idx = np.argsort(np.abs(factor_loadings))[::-1]
        
        # Create gene dataframe
        gene_df = pd.DataFrame({
            'gene': gene_names_filtered[abs_sorted_idx[:100]],
            'loading': factor_loadings[abs_sorted_idx[:100]],
            'abs_loading': np.abs(factor_loadings[abs_sorted_idx[:100]])
        })
        
        gene_df.to_csv(f'gene_loadings_{cell_type}_{region}_{best_factor}.csv', index=False)
    
    return {
        'cell_type': cell_type,
        'region': region,
        'n_animals': n_animals,
        'n_genes': n_genes,
        'best_factor': best_factor,
        'best_age_corr': best_corr,
        'best_pval': best_pval
    }

# Run analysis for all cell types across regions
all_results = []

for cell_type, adata_path in cell_types.items():
    print(f"\n{'#'*70}")
    print(f"CELL TYPE: {cell_type}")
    print(f"{'#'*70}")
    
    # Load to get available regions
    adata = sc.read_h5ad(adata_path, backed='r')
    regions = adata.obs['region'].astype(str).unique()
    print(f"Available regions: {sorted(regions)}")
    
    for region in sorted(regions):
        try:
            result = analyze_celltype_region(cell_type, adata_path, region)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {cell_type} - {region}: {e}")

# Create summary
print(f"\n{'='*70}")
print("SUMMARY ACROSS ALL CELL TYPES")
print(f"{'='*70}")

if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Only keep significant results
    sig_results = results_df[results_df['best_pval'] < 0.05].copy()
    sig_results = sig_results.sort_values('best_age_corr', key=abs, ascending=False)
    
    print(f"\nSignificant age associations (p < 0.05): {len(sig_results)}/{len(results_df)}")
    print("\nTop age-associated factors across all cell types:")
    print(sig_results[['cell_type', 'region', 'best_factor', 'best_age_corr', 'best_pval']].to_string(index=False))
    
    results_df.to_csv('factor_analysis_all_celltypes_summary.csv', index=False)
    sig_results.to_csv('factor_analysis_all_celltypes_significant.csv', index=False)
    print("\nSaved: factor_analysis_all_celltypes_summary.csv")
    print("Saved: factor_analysis_all_celltypes_significant.csv")
    
    # Summary by cell type
    print("\n" + "="*70)
    print("SUMMARY BY CELL TYPE")
    print("="*70)
    for cell_type in sig_results['cell_type'].unique():
        ct_df = sig_results[sig_results['cell_type'] == cell_type]
        print(f"\n{cell_type}:")
        print(f"  Regions with significant age signals: {len(ct_df)}")
        print(f"  Strongest: {ct_df.iloc[0]['region']} (r={ct_df.iloc[0]['best_age_corr']:.3f})")
        print(f"  Regions: {', '.join(ct_df['region'].tolist())}")

print("\nAnalysis complete!")

