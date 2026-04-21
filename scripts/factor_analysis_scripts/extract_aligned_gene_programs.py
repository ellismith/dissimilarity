#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import scanpy as sc

print("="*70)
print("EXTRACTING GENE LOADINGS FROM ALIGNED PCA & FA")
print("="*70)

# We need to rerun PCA and FA to get the components/loadings
# (The saved CSV only has scores, not loadings)

print("\nLoading and preparing data...")

# Load all data (same as before)
all_data = []
all_metadata = []

cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
paths = {
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad'
}

# Track gene names
gene_names = None

for cell_type in cell_types:
    print(f"  Loading {cell_type}...")
    
    adata = sc.read_h5ad(paths[cell_type], backed='r')
    
    # Get gene names from first cell type
    if gene_names is None:
        gene_names = adata.var_names.to_list()
    
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex']].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    obs_df['sex'] = obs_df['sex'].astype(str)
    obs_df['cell_type'] = cell_type
    
    unique_combos = obs_df[['animal_id', 'region', 'age', 'sex', 'cell_type']].drop_duplicates()
    
    for idx, row in unique_combos.iterrows():
        mask = (obs_df['animal_id'] == row['animal_id']) & (obs_df['region'] == row['region'])
        cell_indices = np.where(mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        pseudobulk = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
        all_data.append(pseudobulk)
        all_metadata.append({
            'animal_id': row['animal_id'],
            'region': row['region'],
            'age': row['age'],
            'sex': row['sex'],
            'cell_type': cell_type
        })

print(f"\nTotal samples: {len(all_data)}")

# Stack and filter
data_matrix = np.vstack(all_data)
metadata_df = pd.DataFrame(all_metadata)

min_samples = int(0.2 * len(all_data))
gene_counts = (data_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples

print(f"Keeping {genes_keep.sum()} genes")

data_filtered = data_matrix[:, genes_keep]
gene_names_filtered = [gene_names[i] for i in range(len(gene_names)) if genes_keep[i]]

# Normalize
cpm = (data_filtered / data_filtered.sum(axis=1, keepdims=True)) * 1e6
data_log = np.log2(cpm + 1)

# Standardize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_log)

print("\n" + "="*70)
print("RUNNING PCA (20 components)")
print("="*70)

pca = PCA(n_components=20, random_state=42)
pca_scores = pca.fit_transform(data_scaled)

# Extract loadings (components_ is shape: n_components × n_genes)
pca_loadings = pca.components_  # Shape: (20, n_genes)

print(f"PCA loadings shape: {pca_loadings.shape}")

# Save top genes for key PCs
key_pcs = [19, 20]  # PC19 and PC20 (0-indexed: 18, 19)

for pc_idx in key_pcs:
    pc_num = pc_idx + 1
    print(f"\nExtracting top genes for PC{pc_num}...")
    
    loadings = pca_loadings[pc_idx, :]
    abs_loadings = np.abs(loadings)
    
    # Get top 200 genes by absolute loading
    top_indices = np.argsort(abs_loadings)[::-1][:200]
    
    gene_df = pd.DataFrame({
        'gene_symbol': [gene_names_filtered[i] for i in top_indices],
        'loading': loadings[top_indices],
        'abs_loading': abs_loadings[top_indices]
    })
    
    filename = f'pca_aligned_gene_loadings_PC{pc_num}.csv'
    gene_df.to_csv(filename, index=False)
    print(f"  Saved: {filename}")
    print(f"  Top 10 genes: {', '.join(gene_df['gene_symbol'].head(10).tolist())}")

print("\n" + "="*70)
print("RUNNING FACTOR ANALYSIS (50 factors)")
print("="*70)

fa = FactorAnalysis(n_components=50, random_state=42, max_iter=1000)
fa_scores = fa.fit_transform(data_scaled)

# Extract loadings (components_ is shape: n_components × n_genes)
fa_loadings = fa.components_  # Shape: (50, n_genes)

print(f"FA loadings shape: {fa_loadings.shape}")

# Save top genes for key factors
key_factors = [26, 42, 48]  # Factor26, 42, 48 (0-indexed: 25, 41, 47)

for factor_idx in key_factors:
    factor_num = factor_idx + 1
    print(f"\nExtracting top genes for Factor{factor_num}...")
    
    loadings = fa_loadings[factor_idx, :]
    abs_loadings = np.abs(loadings)
    
    # Get top 200 genes by absolute loading
    top_indices = np.argsort(abs_loadings)[::-1][:200]
    
    gene_df = pd.DataFrame({
        'gene_symbol': [gene_names_filtered[i] for i in top_indices],
        'loading': loadings[top_indices],
        'abs_loading': abs_loadings[top_indices]
    })
    
    filename = f'fa_aligned_gene_loadings_Factor{factor_num}.csv'
    gene_df.to_csv(filename, index=False)
    print(f"  Saved: {filename}")
    print(f"  Top 10 genes: {', '.join(gene_df['gene_symbol'].head(10).tolist())}")

print("\n" + "="*70)
print("Gene extraction complete!")
print("="*70)

