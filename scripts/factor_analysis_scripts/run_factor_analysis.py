#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("Loading pseudobulk data...")
# We'll reload and process the same way as before
import scanpy as sc

adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

# Create pseudobulk
print("Creating pseudobulk...")
obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
obs_df['animal_id'] = obs_df['animal_id'].astype(str)
obs_df['region'] = obs_df['region'].astype(str)
obs_df['sex'] = obs_df['sex'].astype(str)
obs_df['pseudobulk_id'] = obs_df['animal_id'] + '_' + obs_df['region']

pseudobulk_samples = obs_df[['pseudobulk_id', 'animal_id', 'region', 'age', 'sex']].drop_duplicates()

pseudobulk_dict = {}
for pb_id in pseudobulk_samples['pseudobulk_id']:
    cell_mask = obs_df['pseudobulk_id'] == pb_id
    cell_indices = np.where(cell_mask)[0]
    pseudobulk_dict[pb_id] = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()

pseudobulk_matrix = np.vstack([pseudobulk_dict[pb_id] for pb_id in pseudobulk_samples['pseudobulk_id']])

# Filter genes
min_samples = int(0.2 * len(pseudobulk_samples))
gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples
pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]

# Log-normalize
cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
pseudobulk_log = np.log2(cpm + 1)

# Standardize
scaler = StandardScaler()
pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)

print(f"Data shape: {pseudobulk_scaled.shape}")

# Run Factor Analysis with different numbers of factors
print("\nRunning Factor Analysis...")
n_factors = 15

fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
factors = fa.fit_transform(pseudobulk_scaled)

print(f"Fitted {n_factors} latent factors")

# Create results dataframe
fa_df = pd.DataFrame(
    factors,
    columns=[f'Factor{i+1}' for i in range(n_factors)]
)
fa_df = pd.concat([pseudobulk_samples.reset_index(drop=True), fa_df], axis=1)

# Save results
fa_df.to_csv('astrocytes_pseudobulk_factor_analysis.csv', index=False)
print("\nSaved: astrocytes_pseudobulk_factor_analysis.csv")

# Correlate factors with metadata
print("\n=== Factor correlations with Age ===")
age_corrs = []
for i in range(n_factors):
    factor_name = f'Factor{i+1}'
    corr, pval = stats.pearsonr(fa_df[factor_name], fa_df['age'])
    age_corrs.append((factor_name, corr, pval))
    print(f"{factor_name}: r = {corr:.3f}, p = {pval:.3e}")

# Sort by absolute correlation
age_corrs_sorted = sorted(age_corrs, key=lambda x: abs(x[1]), reverse=True)
print("\n=== Top factors by age correlation ===")
for factor, corr, pval in age_corrs_sorted[:5]:
    print(f"{factor}: r = {corr:.3f}, p = {pval:.3e}")

# Check correlations with sex
print("\n=== Factor associations with Sex (t-test) ===")
sex_associations = []
for i in range(n_factors):
    factor_name = f'Factor{i+1}'
    male_vals = fa_df[fa_df['sex'] == 'M'][factor_name]
    female_vals = fa_df[fa_df['sex'] == 'F'][factor_name]
    t_stat, pval = stats.ttest_ind(male_vals, female_vals)
    sex_associations.append((factor_name, t_stat, pval))
    if pval < 0.01:
        print(f"{factor_name}: t = {t_stat:.3f}, p = {pval:.3e}")

# Plot top age-associated factors
print("\nCreating plots...")
top_factors = [x[0] for x in age_corrs_sorted[:4]]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, factor in enumerate(top_factors):
    corr, pval = age_corrs_sorted[idx][1], age_corrs_sorted[idx][2]
    axes[idx].scatter(fa_df['age'], fa_df[factor], alpha=0.5, s=30)
    axes[idx].set_xlabel('Age (years)')
    axes[idx].set_ylabel(factor)
    axes[idx].set_title(f'{factor} vs Age\n(r = {corr:.3f}, p = {pval:.3e})')
    
    # Add regression line
    z = np.polyfit(fa_df['age'], fa_df[factor], 1)
    p = np.poly1d(z)
    axes[idx].plot(fa_df['age'].sort_values(), p(fa_df['age'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('factor_analysis_age_associations.png', dpi=300, bbox_inches='tight')
print("Saved: factor_analysis_age_associations.png")

print("\n=== Comparison with PCA ===")
print("Factor Analysis finds latent factors that may better capture")
print("biological variation compared to PCA's variance-maximizing approach.")
print("\nNext step: Compare these factors to PCA results to see if FA")
print("finds stronger or clearer age associations.")

