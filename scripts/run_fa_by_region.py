#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scanpy as sc

print("Loading data...")
adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

# Get metadata
obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
obs_df['animal_id'] = obs_df['animal_id'].astype(str)
obs_df['region'] = obs_df['region'].astype(str)
obs_df['sex'] = obs_df['sex'].astype(str)

# Get list of regions
regions = obs_df['region'].unique()
print(f"\nAnalyzing {len(regions)} regions: {sorted(regions)}")

# Store results for each region
all_results = []

for region in sorted(regions):
    print(f"\n{'='*60}")
    print(f"Region: {region}")
    print(f"{'='*60}")
    
    # Filter to this region
    region_mask = obs_df['region'] == region
    region_obs = obs_df[region_mask].copy()
    region_obs['pseudobulk_id'] = region_obs['animal_id']  # Just aggregate by animal within region
    
    # Get unique animals in this region
    animal_samples = region_obs[['pseudobulk_id', 'animal_id', 'region', 'age', 'sex']].drop_duplicates()
    n_animals = len(animal_samples)
    print(f"Number of animals: {n_animals}")
    
    if n_animals < 20:
        print(f"Skipping {region} - too few animals")
        continue
    
    # Create pseudobulk by animal
    print("Aggregating by animal...")
    pseudobulk_dict = {}
    cell_indices_region = np.where(region_mask)[0]
    
    for animal_id in animal_samples['animal_id']:
        animal_mask = region_obs['animal_id'] == animal_id
        cell_mask_full = region_mask.copy()
        cell_mask_full[region_mask] = animal_mask.values
        cell_indices = np.where(cell_mask_full)[0]
        pseudobulk_dict[animal_id] = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
    
    pseudobulk_matrix = np.vstack([pseudobulk_dict[aid] for aid in animal_samples['animal_id']])
    
    # Filter genes: keep genes expressed in at least 20% of animals
    min_animals = int(0.2 * n_animals)
    gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
    genes_keep = gene_counts >= min_animals
    n_genes = genes_keep.sum()
    print(f"Keeping {n_genes} genes expressed in ≥{min_animals} animals")
    
    if n_genes < 1000:
        print(f"Skipping {region} - too few genes")
        continue
    
    pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]
    
    # Log-normalize
    cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
    pseudobulk_log = np.log2(cpm + 1)
    
    # Standardize
    scaler = StandardScaler()
    pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)
    
    # Run Factor Analysis
    n_factors = min(10, n_animals - 5)  # Adjust based on sample size
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
    
    for i in range(n_factors):
        factor_name = f'Factor{i+1}'
        corr, pval = stats.pearsonr(fa_df[factor_name], fa_df['age'])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_factor = factor_name
        if pval < 0.05:
            print(f"{factor_name}: r = {corr:.3f}, p = {pval:.3e} ***")
        elif abs(corr) > 0.15:
            print(f"{factor_name}: r = {corr:.3f}, p = {pval:.3e}")
    
    print(f"\nBest age correlation: {best_factor} (r = {best_corr:.3f})")
    
    # Store results
    all_results.append({
        'region': region,
        'n_animals': n_animals,
        'n_genes': n_genes,
        'best_factor': best_factor,
        'best_age_corr': best_corr
    })
    
    # Save region-specific results
    fa_df.to_csv(f'factor_analysis_{region}.csv', index=False)

# Summary across regions
print(f"\n{'='*60}")
print("SUMMARY ACROSS REGIONS")
print(f"{'='*60}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('best_age_corr', key=abs, ascending=False)
print("\nRegions ranked by strongest age correlation:")
print(results_df.to_string(index=False))

results_df.to_csv('factor_analysis_by_region_summary.csv', index=False)
print("\nSaved: factor_analysis_by_region_summary.csv")

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if corr > 0 else 'red' for corr in results_df['best_age_corr']]
bars = ax.barh(results_df['region'], results_df['best_age_corr'], color=colors, alpha=0.7)
ax.set_xlabel('Best Age Correlation (r)')
ax.set_ylabel('Region')
ax.set_title('Strongest Age-Factor Correlation by Brain Region')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('age_correlation_by_region.png', dpi=300, bbox_inches='tight')
print("Saved: age_correlation_by_region.png")

print("\nAnalysis complete!")

