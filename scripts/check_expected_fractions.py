import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"

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
louvain_clusters = np.unique(louvain[region_mask])

summary = []

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < 10:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    unique_animals = np.unique(animals_in_cluster)
    donor_props = [(animals_in_cluster == a).mean() for a in unique_animals]

    summary.append({
        "louvain": cl,
        "n_cells": len(cell_indices),
        "min_exp_frac": np.min(donor_props),
        "median_exp_frac": np.median(donor_props),
        "n_donors": len(unique_animals)
    })

df = pd.DataFrame(summary).sort_values("min_exp_frac")
out = "/scratch/easmit31/factor_analysis/expected_fraction_summary.csv"
df.to_csv(out, index=False)

print(df.head(10).to_string())
print(f"\nSaved to {out}")
