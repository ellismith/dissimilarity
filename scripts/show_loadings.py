#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import scanpy as sc

print("="*70)
print("SHOWING PCA & FA LOADINGS")
print("="*70)

# Load data
print("\nLoading data...")

all_data = []
cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
paths = {
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad'
}

gene_names = None

for cell_type in cell_types:
    print(f"  {cell_type}...", end='')
    
    adata = sc.read_h5ad(paths[cell_type], backed='r')
    
    if gene_names is None:
        gene_names = adata.var_names.to_list()
    
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    obs_df['cell_type'] = cell_type
    
    unique_combos = obs_df[['animal_id', 'region']].drop_duplicates()
    
    for idx, row in unique_combos.iterrows():
        mask = (obs_df['animal_id'] == row['animal_id']) & (obs_df['region'] == row['region'])
        cell_indices = np.where(mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        pseudobulk = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
        all_data.append(pseudobulk)
    
    print(" done")

data_matrix = np.vstack(all_data)

# Filter genes
min_samples = int(0.2 * len(all_data))
gene_counts = (data_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples

data_filtered = data_matrix[:, genes_keep]
gene_names_filtered = [gene_names[i] for i in range(len(gene_names)) if genes_keep[i]]

print(f"\nSamples: {len(all_data)}, Genes: {len(gene_names_filtered)}")

# Normalize & scale
cpm = (data_filtered / data_filtered.sum(axis=1, keepdims=True)) * 1e6
data_log = np.log2(cpm + 1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_log)

# ==================== PCA ====================
print("\n" + "="*70)
print("PCA LOADINGS")
print("="*70)

pca = PCA(n_components=20, random_state=42)
pca.fit(data_scaled)

print("\nPCA components_ shape:", pca.components_.shape)
print("(rows = components, columns = genes)")

# Show PC19 and PC20 loadings
for pc_idx in [18, 19]:  # 0-indexed
    pc_num = pc_idx + 1
    print(f"\n{'='*70}")
    print(f"PC{pc_num} LOADINGS")
    print(f"{'='*70}")
    
    loadings = pca.components_[pc_idx, :]
    abs_loadings = np.abs(loadings)
    
    # Top genes
    top_idx = np.argsort(abs_loadings)[::-1][:20]
    
    print(f"\nTop 20 genes by |loading|:")
    print(f"{'Gene':<15} {'Loading':>10} {'|Loading|':>10}")
    print("-"*40)
    for i in top_idx:
        print(f"{gene_names_filtered[i]:<15} {loadings[i]:>10.4f} {abs_loadings[i]:>10.4f}")
    
    # Stats
    print(f"\nLoading statistics:")
    print(f"  Mean: {loadings.mean():.6f}")
    print(f"  Std: {loadings.std():.6f}")
    print(f"  Min: {loadings.min():.6f}")
    print(f"  Max: {loadings.max():.6f}")
    print(f"  Mean |loading|: {abs_loadings.mean():.6f}")

# ==================== FA ====================
print("\n" + "="*70)
print("FACTOR ANALYSIS LOADINGS")
print("="*70)

fa = FactorAnalysis(n_components=50, random_state=42, max_iter=1000)
fa.fit(data_scaled)

print("\nFA components_ shape:", fa.components_.shape)
print("(rows = factors, columns = genes)")

# Show Factor26, 42, 48 loadings
for factor_idx in [25, 41, 47]:  # 0-indexed
    factor_num = factor_idx + 1
    print(f"\n{'='*70}")
    print(f"FACTOR{factor_num} LOADINGS")
    print(f"{'='*70}")
    
    loadings = fa.components_[factor_idx, :]
    abs_loadings = np.abs(loadings)
    
    # Top genes
    top_idx = np.argsort(abs_loadings)[::-1][:20]
    
    print(f"\nTop 20 genes by |loading|:")
    print(f"{'Gene':<15} {'Loading':>10} {'|Loading|':>10}")
    print("-"*40)
    for i in top_idx:
        print(f"{gene_names_filtered[i]:<15} {loadings[i]:>10.4f} {abs_loadings[i]:>10.4f}")
    
    # Stats
    print(f"\nLoading statistics:")
    print(f"  Mean: {loadings.mean():.6f}")
    print(f"  Std: {loadings.std():.6f}")
    print(f"  Min: {loadings.min():.6f}")
    print(f"  Max: {loadings.max():.6f}")
    print(f"  Mean |loading|: {abs_loadings.mean():.6f}")

print("\n" + "="*70)
print("Done!")
print("="*70)

