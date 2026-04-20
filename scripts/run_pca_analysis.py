#!/usr/bin/env python

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

print(f"Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

# Create pseudobulk by animal_id x region
print("\nCreating pseudobulk aggregation by animal_id × region...")

# Get the metadata we need - convert categoricals to strings
obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
obs_df['animal_id'] = obs_df['animal_id'].astype(str)
obs_df['region'] = obs_df['region'].astype(str)
obs_df['sex'] = obs_df['sex'].astype(str)
obs_df['pseudobulk_id'] = obs_df['animal_id'] + '_' + obs_df['region']

# Get unique pseudobulk samples
pseudobulk_samples = obs_df[['pseudobulk_id', 'animal_id', 'region', 'age', 'sex']].drop_duplicates()
print(f"Number of pseudobulk samples: {len(pseudobulk_samples)}")

# Aggregate expression by pseudobulk_id
print("Aggregating expression data...")
pseudobulk_dict = {}
for pb_id in pseudobulk_samples['pseudobulk_id']:
    cell_mask = obs_df['pseudobulk_id'] == pb_id
    cell_indices = np.where(cell_mask)[0]
    # Sum counts for cells in this pseudobulk sample
    pseudobulk_dict[pb_id] = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()

# Create pseudobulk matrix
pseudobulk_matrix = np.vstack([pseudobulk_dict[pb_id] for pb_id in pseudobulk_samples['pseudobulk_id']])
print(f"Pseudobulk matrix shape: {pseudobulk_matrix.shape}")

# Filter genes: keep genes expressed in at least 20% of pseudobulk samples
min_samples = int(0.2 * len(pseudobulk_samples))
gene_counts = (pseudobulk_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples
print(f"Keeping {genes_keep.sum()} genes expressed in ≥{min_samples} samples")

pseudobulk_filtered = pseudobulk_matrix[:, genes_keep]
gene_names_filtered = adata.var_names[genes_keep]

# Log-normalize: log2(CPM + 1)
print("\nLog-normalizing...")
cpm = (pseudobulk_filtered / pseudobulk_filtered.sum(axis=1, keepdims=True)) * 1e6
pseudobulk_log = np.log2(cpm + 1)

# Standardize for PCA
print("Standardizing...")
scaler = StandardScaler()
pseudobulk_scaled = scaler.fit_transform(pseudobulk_log)

# Run PCA
print("\nRunning PCA...")
pca = PCA(n_components=20)
pca_result = pca.fit_transform(pseudobulk_scaled)

# Create results dataframe
pca_df = pd.DataFrame(
    pca_result,
    columns=[f'PC{i+1}' for i in range(20)]
)
pca_df = pd.concat([pseudobulk_samples.reset_index(drop=True), pca_df], axis=1)

# Save results
print("\nSaving results...")
pca_df.to_csv('astrocytes_pseudobulk_pca.csv', index=False)

# Print variance explained
print("\n=== Variance explained by each PC ===")
for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"Cumulative variance (PC1-10): {pca.explained_variance_ratio_[:10].sum():.4f}")

# Correlate PCs with metadata
print("\n=== Correlation of PCs with metadata ===")
for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
    age_corr = np.corrcoef(pca_df[pc], pca_df['age'])[0, 1]
    print(f"{pc} vs Age: r = {age_corr:.3f}")

# Plot PC1 vs PC2
print("\nCreating plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Color by age
scatter1 = axes[0].scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['age'], cmap='viridis', s=50, alpha=0.7)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title('PCA colored by Age')
plt.colorbar(scatter1, ax=axes[0], label='Age (years)')

# Color by sex
for sex, color in zip(['F', 'M'], ['red', 'blue']):
    mask = pca_df['sex'] == sex
    axes[1].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                   c=color, label=sex, s=50, alpha=0.7)
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_title('PCA colored by Sex')
axes[1].legend()

# Color by region
regions = pca_df['region'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
for region, color in zip(regions, colors):
    mask = pca_df['region'] == region
    axes[2].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                   c=[color], label=region, s=50, alpha=0.7)
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[2].set_title('PCA colored by Region')
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('astrocytes_pca_plots.png', dpi=300, bbox_inches='tight')
print("Saved: astrocytes_pca_plots.png")

print("\nAnalysis complete!")
