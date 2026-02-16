import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import scipy.sparse as sp

# Load louvain 1 HIP data
adata = sc.read_h5ad('/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad', backed='r')

mask = (adata.obs['louvain'] == '1') & (adata.obs['region'] == 'HIP') & (adata.obs['age'] >= 1)
cell_indices = np.where(mask)[0]

X_subset = adata.X[cell_indices, :]

# Filter genes
gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
pct_expressing = gene_expressed / len(cell_indices)
genes_pass = pct_expressing >= 0.05
X_filtered = X_subset[:, genes_pass]
X_dense = X_filtered.toarray()

# Get metadata
metadata = adata.obs.iloc[cell_indices].copy().reset_index(drop=True)

# Pick cell 100
cell_idx = 100

print(f"Analyzing cell {cell_idx}")
print(f"Animal: {metadata.iloc[cell_idx]['animal_id']}")
print(f"Age: {metadata.iloc[cell_idx]['age']:.2f}")
print(f"\n{'='*70}\n")

# Compute distances with each metric
k = 10
results = {}

for name, X, metric in [
    ('Raw_Euclidean', X_dense, 'euclidean'),
    ('Raw_Cosine', X_dense, 'cosine'),
]:
    print(f"{name}:")
    dist = pairwise_distances(X, metric=metric)
    
    d = dist[cell_idx, :].copy()
    d[cell_idx] = np.inf
    
    # Get top 10
    knn_indices = np.argsort(d)[:k]
    results[name] = set(knn_indices)
    
    print(f"  Neighbors: {knn_indices}")
    print(f"  Animals: {metadata.iloc[knn_indices]['animal_id'].values}")
    print(f"  Ages: {metadata.iloc[knn_indices]['age'].values}")
    print()

# Check overlap
overlap = results['Raw_Euclidean'] & results['Raw_Cosine']
print(f"Overlap: {len(overlap)} / {k} neighbors")
print(f"Shared indices: {overlap}")
print(f"\nEuclidean-only: {results['Raw_Euclidean'] - results['Raw_Cosine']}")
print(f"Cosine-only: {results['Raw_Cosine'] - results['Raw_Euclidean']}")
