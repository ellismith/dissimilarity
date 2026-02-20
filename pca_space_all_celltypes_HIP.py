import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

region = "HIP"
n_pcs = 30

cell_type_files = {
    'astrocytes': 'Res1_astrocytes_update.h5ad',
    'ependymal-cells': 'Res1_ependymal-cells_new.h5ad',
    'GABAergic-neurons': 'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons': 'Res1_glutamatergic-neurons_update.h5ad',
    'microglia': 'Res1_microglia_new.h5ad',
    'opc-olig': 'Res1_opc-olig_subset.h5ad',
    'vascular-cells': 'Res1_vascular-cells_subset.h5ad',
}

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

for cell_type, fname in cell_type_files.items():
    fpath = f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{fname}'
    print(f"Processing {cell_type}...")

    try:
        with h5py.File(fpath, 'r') as f:
            X_pca = f['obsm']['X_pca'][:, :n_pcs]
            regions = decode_categorical(f['obs']['region'])
            louvain = decode_categorical(f['obs']['louvain'])
            animal_ids = decode_categorical(f['obs']['animal_id'])
            ages = f['obs']['age'][:]
    except Exception as e:
        print(f"  Error: {e}, skipping")
        continue

    region_mask = (regions == region) & (ages >= 1)
    if region_mask.sum() == 0:
        print(f"  No cells in {region}, skipping")
        continue

    louvain_clusters = np.unique(louvain[region_mask])

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

        sc = axes[0].scatter(pca_subset[:, 0], pca_subset[:, 1], c=distances, cmap='viridis', s=2, alpha=0.5)
        axes[0].scatter(centroid[0], centroid[1], color='red', s=100, marker='x', zorder=5, label='centroid')
        plt.colorbar(sc, ax=axes[0], label='dist to centroid')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title(f'{cell_type} {region} louvain {cl} - distance')
        axes[0].legend()

        sc2 = axes[1].scatter(pca_subset[:, 0], pca_subset[:, 1], c=ages_subset, cmap='plasma', s=2, alpha=0.5)
        axes[1].scatter(centroid[0], centroid[1], color='red', s=100, marker='x', zorder=5, label='centroid')
        plt.colorbar(sc2, ax=axes[1], label='age')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title(f'{cell_type} {region} louvain {cl} - age')
        axes[1].legend()

        plt.tight_layout()
        out = f'/scratch/easmit31/factor_analysis/{cell_type}_{region}_louvain{cl}_pcspace.png'
        plt.savefig(out, dpi=150)
        plt.close()

    print(f"  Done {cell_type}")

print("All done.")
