import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cell_class_file = "Res1_GABAergic-neurons_subset.h5ad"
region = "HIP"
n_pcs = 30

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

with h5py.File(f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{cell_class_file}', 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    ages = f['obs']['age'][:]

region_mask = (regions == region) & (ages >= 1)
louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {region}")

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    if mask.sum() < 10:
        continue

    pca_subset = X_pca[mask]
    ages_subset = ages[mask]
    centroid = pca_subset.mean(axis=0)
    diffs = pca_subset - centroid
    distances = np.sqrt((diffs ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(pca_subset[:, 0], pca_subset[:, 1], c=distances, cmap='viridis', s=3, alpha=0.6)
    axes[0].scatter(centroid[0], centroid[1], color='red', s=100, marker='x', zorder=5, label='centroid')
    plt.colorbar(sc, ax=axes[0], label='dist to centroid')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title(f'GABAergic {region} louvain {cl} - distance')
    axes[0].legend()

    sc2 = axes[1].scatter(pca_subset[:, 0], pca_subset[:, 1], c=ages_subset, cmap='plasma', s=3, alpha=0.6)
    axes[1].scatter(centroid[0], centroid[1], color='red', s=100, marker='x', zorder=5, label='centroid')
    plt.colorbar(sc2, ax=axes[1], label='age')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'GABAergic {region} louvain {cl} - age')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{cl}_pcspace.png', dpi=150)
    plt.close()
    print(f"Saved louvain {cl} (n cells={mask.sum()})")

print("Done.")
