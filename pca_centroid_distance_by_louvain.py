"""
pca_centroid_distance_by_louvain.py

For each louvain cluster in a given cell type and brain region, computes
per-animal centroids in PCA space, then measures each cell's distance to
its own animal's centroid. This captures within-animal transcriptional
spread rather than deviation from a population mean.

Two analyses per louvain:
1. Mean of per-cell distances per animal ~ age
   (do cells sit further from their own animal's centroid with age?)
2. Variance of per-cell distances per animal ~ age
   (does within-animal transcriptional heterogeneity increase with age?)

Input:
    - Cell-class h5ad file (read via h5py)
    - PCA coordinates stored in obsm['X_pca']

Output:
    - Two scatter plots per louvain (mean and variance vs age)
    - Saved to /scratch/easmit31/factor_analysis/

Usage:
    python pca_centroid_distance_by_louvain.py
    (edit parameters at top of script to change cell type, region, etc.)
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Parameters ---
cell_class_file = "Res1_GABAergic-neurons_subset.h5ad"
region = "HIP"
n_pcs = 50      # number of PCs to use for distance calculation
min_cells = 100  # skip louvain clusters with fewer total cells than this
min_age = 1.0   # exclude animals younger than this
min_cells_per_animal = 2  # minimum cells per animal to compute a centroid

# --- Helper functions ---
def decode_categorical(grp):
    """Decode an h5py categorical group (categories + codes) to numpy string array."""
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def decode_col(grp):
    """Decode a plain h5py string dataset to numpy string array."""
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

# --- Load data ---
fpath = f'/data/CEM/smacklab/U01/{cell_class_file}'
print(f"Loading {fpath}...")
with h5py.File(fpath, 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    barcodes = decode_col(f['obs']['_index'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages = f['obs']['age'][:]

# --- Filter to region and min age ---
region_mask = (regions == region) & (ages >= min_age)
louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {region}")

# --- Main loop over louvain clusters ---
for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    if mask.sum() < min_cells:
        continue

    cell_indices = np.where(mask)[0]

    # For each animal, compute that animal's centroid and each cell's distance to it
    rows = []
    for animal in np.unique(animal_ids[cell_indices]):
        animal_mask = mask & (animal_ids == animal)
        idxs = np.where(animal_mask)[0]
        if len(idxs) < min_cells_per_animal:
            continue
        local_pca = X_pca[idxs]
        animal_centroid = local_pca.mean(axis=0)  # centroid for this animal only
        dists = np.sqrt(((local_pca - animal_centroid) ** 2).sum(axis=1))
        age_val = ages[idxs[0]]
        for d in dists:
            rows.append({'animal_id': animal, 'age': age_val, 'dist_to_centroid': d})

    df = pd.DataFrame(rows)
    if df.empty:
        continue

    # Aggregate mean and variance of distances per animal
    agg_mean = df.groupby(['animal_id', 'age'])['dist_to_centroid'].mean().reset_index().sort_values('age')
    agg_var  = df.groupby(['animal_id', 'age'])['dist_to_centroid'].var().reset_index().sort_values('age')
    agg_var  = agg_var.dropna(subset=['dist_to_centroid'])

    if len(agg_mean) < 5 or len(agg_var) < 5:
        continue

    # Linear regression for both
    slope_m, intercept_m, r_m, p_m, _ = stats.linregress(agg_mean['age'], agg_mean['dist_to_centroid'])
    slope_v, intercept_v, r_v, p_v, _ = stats.linregress(agg_var['age'],  agg_var['dist_to_centroid'])

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, agg, slope, intercept, r, p, ylabel in [
        (axes[0], agg_mean, slope_m, intercept_m, r_m, p_m, 'Mean dist to animal centroid'),
        (axes[1], agg_var,  slope_v, intercept_v, r_v, p_v, 'Variance of dist to animal centroid'),
    ]:
        ax.scatter(agg['age'], agg['dist_to_centroid'], color='steelblue', s=40)
        x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='firebrick', lw=1.5)
        ax.set_xlabel('Age')
        ax.set_ylabel(ylabel)
        ax.set_title(f'GABAergic {region} louvain {cl}\nr={r:.2f}, p={p:.3f}, n={len(agg)} animals')

    plt.tight_layout()
    out = f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{cl}_animal_centroid_mean_var.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved louvain {cl} (n cells={mask.sum()}, n animals={len(agg_mean)}, mean p={p_m:.3f}, var p={p_v:.3f})")

print("Done.")
