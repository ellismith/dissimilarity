#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scanpy as sc

print("="*70)
print("COMBINED PCA WITH 50 COMPONENTS")
print("="*70)

# Load all data (reusing previous logic)
all_data = []
all_metadata = []

cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
paths = {
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad'
}

for cell_type in cell_types:
    print(f"\nLoading {cell_type}...")
    
    adata = sc.read_h5ad(paths[cell_type], backed='r')
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

# Stack into matrix
data_matrix = np.vstack(all_data)
metadata_df = pd.DataFrame(all_metadata)

# Filter genes
min_samples = int(0.2 * len(all_data))
gene_counts = (data_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples
n_genes = genes_keep.sum()

print(f"Keeping {n_genes} genes")

data_filtered = data_matrix[:, genes_keep]

# Normalize
cpm = (data_filtered / data_filtered.sum(axis=1, keepdims=True)) * 1e6
data_log = np.log2(cpm + 1)

# Standardize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_log)

# Run PCA with 50 components
print("\nRunning PCA with 50 components...")
pca = PCA(n_components=50, random_state=42)
pc_scores = pca.fit_transform(data_scaled)

# Create results dataframe
pca_df = pd.DataFrame(
    pc_scores,
    columns=[f'PC{i+1}' for i in range(50)]
)
pca_df = pd.concat([metadata_df.reset_index(drop=True), pca_df], axis=1)

# Save
pca_df.to_csv('pca_combined_all_celltypes_50pcs.csv', index=False)
print("Saved: pca_combined_all_celltypes_50pcs.csv")

# Quick interpretation
print("\n" + "="*70)
print("QUICK INTERPRETATION (PC21-50)")
print("="*70)

cumulative_var = 0
for i in range(50):
    cumulative_var += pca.explained_variance_ratio_[i]

print(f"\nPC1-50 explain {cumulative_var*100:.1f}% of variance")

# Check for age signals in later PCs
age_pcs = []
for i in range(1, 51):
    pc = f'PC{i}'
    r, p = stats.pearsonr(pca_df['age'], pca_df[pc])
    if p < 0.05:
        age_pcs.append((i, r, p))

print(f"\nAge-associated PCs (p<0.05 overall): {len(age_pcs)}")
print("\nLater PCs (21-50) with age effects:")
for pc_num, r, p in age_pcs:
    if pc_num >= 21:
        print(f"  PC{pc_num}: r={r:.3f}, p={p:.3e}")

print("\n" + "="*70)
print("Now run test_age_by_group.py with 50 PCs!")
print("="*70)

