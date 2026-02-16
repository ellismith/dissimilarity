import numpy as np
from sklearn.metrics import pairwise_distances
import scanpy as sc
import scipy.sparse as sp

# Load data
adata = sc.read_h5ad('/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad', backed='r')
mask = (adata.obs['louvain'] == '1') & (adata.obs['region'] == 'HIP') & (adata.obs['age'] >= 1)
cell_indices = np.where(mask)[0]
X_subset = adata.X[cell_indices, :]
gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
pct_expressing = gene_expressed / len(cell_indices)
genes_pass = pct_expressing >= 0.05
X_filtered = X_subset[:, genes_pass]
X_dense = X_filtered.toarray()

# For cell 100, look at ALL distances
cell_idx = 100

for name, metric in [('Euclidean', 'euclidean'), ('Cosine', 'cosine')]:
    dist = pairwise_distances(X_dense, metric=metric)
    d = dist[cell_idx, :].copy()
    d[cell_idx] = np.inf
    
    # Sort distances
    sorted_idx = np.argsort(d)
    sorted_dist = d[sorted_idx]
    
    print(f"\n{name}:")
    print(f"  Distance to NN #1: {sorted_dist[0]:.4f}")
    print(f"  Distance to NN #10: {sorted_dist[9]:.4f}")
    print(f"  Distance to NN #20: {sorted_dist[19]:.4f}")
    print(f"  Distance to NN #50: {sorted_dist[49]:.4f}")
    print(f"  Distance to NN #100: {sorted_dist[99]:.4f}")
    print(f"\n  Difference between NN#10 and NN#11: {sorted_dist[10] - sorted_dist[9]:.4f}")
    print(f"  Difference between NN#1 and NN#10: {sorted_dist[9] - sorted_dist[0]:.4f}")
    print(f"  % difference (NN#10 vs NN#1): {(sorted_dist[9] - sorted_dist[0])/sorted_dist[0] * 100:.1f}%")
