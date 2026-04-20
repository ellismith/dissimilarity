#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scanpy as sc

print("="*70)
print("COMBINED PCA ACROSS ALL CELL TYPES/REGIONS")
print("="*70)

# For each cell type, load all regions and stack
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
    obs_df['cell_type'] = cell_type
    
    # Get unique animal-region combinations
    unique_combos = obs_df[['animal_id', 'region', 'age', 'sex', 'cell_type']].drop_duplicates()
    
    print(f"  {len(unique_combos)} animal-region combinations")
    
    # Create pseudobulk for each
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

print(f"Matrix shape: {data_matrix.shape}")

# Filter genes (expressed in 20% of samples)
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

# Run PCA
print("\nRunning PCA with 20 components...")
pca = PCA(n_components=20, random_state=42)
pc_scores = pca.fit_transform(data_scaled)

# Create results dataframe
pca_df = pd.DataFrame(
    pc_scores,
    columns=[f'PC{i+1}' for i in range(20)]
)
pca_df = pd.concat([metadata_df.reset_index(drop=True), pca_df], axis=1)

# Save
pca_df.to_csv('pca_combined_all_celltypes.csv', index=False)
print("Saved: pca_combined_all_celltypes.csv")

# Analyze each PC
print("\n" + "="*70)
print("PC INTERPRETATIONS")
print("="*70)

for i in range(20):
    pc = f'PC{i+1}'
    var_exp = pca.explained_variance_ratio_[i]
    
    print(f"\n{pc} ({var_exp*100:.1f}% variance):")
    
    # Age
    r_age, p_age = stats.pearsonr(pca_df['age'], pca_df[pc])
    if p_age < 0.05:
        print(f"  ✓ AGE: r={r_age:.3f}, p={p_age:.3e}")
    
    # Sex
    from scipy.stats import ttest_ind
    males = pca_df[pca_df['sex'] == 'M'][pc]
    females = pca_df[pca_df['sex'] == 'F'][pc]
    t, p_sex = ttest_ind(males, females)
    if p_sex < 0.05:
        print(f"  ✓ SEX: p={p_sex:.3e}")
    
    # Cell type
    from scipy.stats import f_oneway
    groups = [pca_df[pca_df['cell_type'] == ct][pc].values for ct in cell_types]
    f, p_ct = f_oneway(*groups)
    if p_ct < 0.05:
        print(f"  ✓ CELL TYPE: p={p_ct:.3e}")
    
    # Region
    regions = pca_df['region'].unique()
    groups = [pca_df[pca_df['region'] == r][pc].values for r in regions if len(pca_df[pca_df['region'] == r]) > 0]
    if len(groups) > 1:
        f, p_reg = f_oneway(*groups)
        if p_reg < 0.05:
            print(f"  ✓ REGION: p={p_reg:.3e}")

print("\n" + "="*70)
print("Analysis complete! Now PC2 means the same thing everywhere.")
print("="*70)

