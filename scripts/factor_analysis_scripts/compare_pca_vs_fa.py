#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scanpy as sc

print("="*70)
print("DIRECT COMPARISON: PCA vs Factor Analysis")
print("="*70)

# Test on a few representative cases
test_cases = [
    ('Glutamatergic', '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad', 'M1'),
    ('GABAergic', '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad', 'dlPFC'),
    ('Astrocytes', '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad', 'CN'),
    ('Microglia', '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad', 'EC'),
]

all_results = []

for cell_type, adata_path, region in test_cases:
    print(f"\n{'='*70}")
    print(f"Testing: {cell_type} - {region}")
    print(f"{'='*70}")
    
    # Load data
    adata = sc.read_h5ad(adata_path, backed='r')
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    
    # Filter to region
    region_mask = obs_df['region'] == region
    region_obs = obs_df[region_mask].copy()
    
    if region_obs.empty:
        continue
    
    # Get unique animals
    animal_samples = region_obs[['animal_id', 'age', 'sex']].drop_duplicates()
    n_animals = len(animal_samples)
    
    print(f"Animals: {n_animals}")
    
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
    min_animals = int(0.2 * n_animals)
    gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
    genes_keep = gene_counts >= min_animals
    n_genes = genes_keep.sum()
    
    print(f"Genes: {n_genes}")
    
    pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]
    
    # Normalize
    cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
    pseudobulk_log = np.log2(cpm + 1)
    scaler = StandardScaler()
    pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)
    
    # === RUN BOTH PCA AND FA ===
    n_components = min(10, n_animals - 5)
    
    # PCA
    print(f"\nRunning PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_scores = pca.fit_transform(pseudobulk_scaled)
    
    # Factor Analysis
    print(f"Running Factor Analysis with {n_components} factors...")
    fa = FactorAnalysis(n_components=n_components, random_state=42, max_iter=1000)
    fa_scores = fa.fit_transform(pseudobulk_scaled)
    
    # === COMPARE AGE CORRELATIONS ===
    age = animal_samples['age'].values
    
    print(f"\n--- PCA Age Correlations ---")
    pca_best_r = 0
    pca_best_pc = None
    pca_best_p = 1
    
    for i in range(n_components):
        r, p = stats.pearsonr(pca_scores[:, i], age)
        if abs(r) > abs(pca_best_r):
            pca_best_r = r
            pca_best_pc = i + 1
            pca_best_p = p
        if p < 0.05:
            print(f"PC{i+1}: r={r:.3f}, p={p:.3e}, var_explained={pca.explained_variance_ratio_[i]:.1%}")
    
    print(f"\nBest PCA: PC{pca_best_pc}, r={pca_best_r:.3f}, p={pca_best_p:.3e}")
    
    print(f"\n--- Factor Analysis Age Correlations ---")
    fa_best_r = 0
    fa_best_f = None
    fa_best_p = 1
    
    for i in range(n_components):
        r, p = stats.pearsonr(fa_scores[:, i], age)
        if abs(r) > abs(fa_best_r):
            fa_best_r = r
            fa_best_f = i + 1
            fa_best_p = p
        if p < 0.05:
            print(f"Factor{i+1}: r={r:.3f}, p={p:.3e}")
    
    print(f"\nBest FA: Factor{fa_best_f}, r={fa_best_r:.3f}, p={fa_best_p:.3e}")
    
    # === COMPARE GENE LOADINGS ===
    # Are the genes similar?
    if pca_best_p < 0.05 or fa_best_p < 0.05:
        pca_loadings = pca.components_[pca_best_pc - 1]
        fa_loadings = fa.components_[fa_best_f - 1]
        
        # Get top 50 genes for each
        pca_top50_idx = np.argsort(np.abs(pca_loadings))[-50:]
        fa_top50_idx = np.argsort(np.abs(fa_loadings))[-50:]
        
        overlap = len(set(pca_top50_idx) & set(fa_top50_idx))
        print(f"\nTop 50 gene overlap: {overlap}/50 ({overlap/50*100:.0f}%)")
        
        # Correlation between loading vectors
        loading_corr, _ = stats.pearsonr(pca_loadings, fa_loadings)
        print(f"Loading vector correlation: r={loading_corr:.3f}")
    
    # Store results
    all_results.append({
        'cell_type': cell_type,
        'region': region,
        'n_animals': n_animals,
        'n_genes': n_genes,
        'pca_best_r': pca_best_r,
        'pca_best_p': pca_best_p,
        'pca_best_component': pca_best_pc,
        'fa_best_r': fa_best_r,
        'fa_best_p': fa_best_p,
        'fa_best_component': fa_best_f,
        'abs_diff': abs(abs(fa_best_r) - abs(pca_best_r)),
        'fa_better': abs(fa_best_r) > abs(pca_best_r)
    })

# Summary
print("\n" + "="*70)
print("SUMMARY: PCA vs Factor Analysis")
print("="*70)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Statistics
print(f"\n--- Overall Comparison ---")
print(f"Cases where FA found stronger signal: {results_df['fa_better'].sum()}/{len(results_df)}")
print(f"Mean |r| for PCA: {results_df['pca_best_r'].abs().mean():.3f}")
print(f"Mean |r| for FA:  {results_df['fa_best_r'].abs().mean():.3f}")
print(f"Mean absolute difference: {results_df['abs_diff'].mean():.3f}")

# Statistical test
if len(results_df) > 0:
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(results_df['fa_best_r'].abs(), results_df['pca_best_r'].abs())
        print(f"Wilcoxon test (FA vs PCA): p={p:.3f}")
    except:
        print("Not enough data for statistical test")

results_df.to_csv('pca_vs_fa_comparison.csv', index=False)
print(f"\nSaved: pca_vs_fa_comparison.csv")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

fa_wins = results_df['fa_better'].sum()
pca_wins = len(results_df) - fa_wins
mean_diff = results_df['abs_diff'].mean()

if fa_wins > pca_wins and mean_diff > 0.05:
    print("✓ Factor Analysis shows consistently stronger age signals")
    print(f"  FA wins: {fa_wins}/{len(results_df)} cases")
    print(f"  Mean improvement: {mean_diff:.3f}")
elif abs(results_df['fa_best_r'].abs().mean() - results_df['pca_best_r'].abs().mean()) < 0.02:
    print("≈ PCA and Factor Analysis perform similarly")
    print("  Difference is negligible - either method would work")
    print("  FA's theoretical advantages (noise modeling) don't materialize here")
else:
    print("? Mixed results - context dependent")

