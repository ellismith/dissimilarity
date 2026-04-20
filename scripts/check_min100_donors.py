import h5py
import numpy as np
import pandas as pd

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
min_cells = 100

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

print("Loading data...")
with h5py.File(fpath, 'r') as f:
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages = f['obs']['age'][:]

region_mask = (regions == region) & (ages >= 1)
clusters = np.unique(louvain[region_mask])

summary = []

for cl in clusters:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) == 0:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    counts = pd.Series(animals_in_cluster).value_counts()

    n_total_donors = len(counts)
    n_ge_100 = (counts >= min_cells).sum()
    max_cells = counts.max()
    median_cells = counts.median()

    summary.append({
        "louvain": cl,
        "cluster_cells": len(cell_indices),
        "total_donors": n_total_donors,
        "donors_ge_100": n_ge_100,
        "median_cells_per_donor": median_cells,
        "max_cells_single_donor": max_cells
    })

df = pd.DataFrame(summary).sort_values("donors_ge_100", ascending=False)

out = "/scratch/easmit31/factor_analysis/min100_donor_summary.csv"
df.to_csv(out, index=False)

print(df.to_string())
print(f"\nSaved to {out}")
