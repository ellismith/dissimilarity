import h5py
import numpy as np
import pandas as pd

cell_class_file = "Res1_GABAergic-neurons_subset.h5ad"
region = "HIP"
louvain_cluster = "1"
n_pcs = 30

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

with h5py.File(f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{cell_class_file}', 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    barcodes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f['obs']['_index'][:]])

mask = (regions == region) & (louvain == louvain_cluster)
print(f"Cells in {region} louvain {louvain_cluster}: {mask.sum()}")

pca_subset = X_pca[mask]
barcodes_subset = barcodes[mask]

centroid = pca_subset.mean(axis=0)
diffs = pca_subset - centroid
distances = np.sqrt((diffs ** 2).sum(axis=1))

df = pd.DataFrame({
    'barcode': barcodes_subset,
    'dist_to_centroid': distances
})

print(df.describe())

out = f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{louvain_cluster}_centroid_distances.csv'
df.to_csv(out, index=False)
print(f"Saved to {out}")
