#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scanpy as sc

print("="*70)
print("COMBINED FACTOR ANALYSIS (ALIGNED AXES)")
print("="*70)

# Load all data (same as PCA)
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

# Run Factor Analysis with 50 factors
print("\nRunning Factor Analysis with 50 factors...")
print("(This may take several minutes...)")

fa = FactorAnalysis(n_components=50, random_state=42, max_iter=1000)
factor_scores = fa.fit_transform(data_scaled)

# Create results dataframe
fa_df = pd.DataFrame(
    factor_scores,
    columns=[f'Factor{i+1}' for i in range(50)]
)
fa_df = pd.concat([metadata_df.reset_index(drop=True), fa_df], axis=1)

# Save
fa_df.to_csv('fa_combined_all_celltypes_50factors.csv', index=False)
print("Saved: fa_combined_all_celltypes_50factors.csv")

# Quick interpretation
print("\n" + "="*70)
print("QUICK INTERPRETATION")
print("="*70)

# Check for age signals
age_factors = []
for i in range(1, 51):
    factor = f'Factor{i}'
    r, p = stats.pearsonr(fa_df['age'], fa_df[factor])
    if p < 0.05:
        age_factors.append((i, r, p))
        if p < 0.001:
            print(f"Factor{i}: r={r:.3f}, p={p:.3e} ***")

print(f"\nTotal factors with age effect (p<0.05): {len(age_factors)}")

print("\n" + "="*70)
print("Analysis complete! Now run age testing by group.")
print("="*70)

