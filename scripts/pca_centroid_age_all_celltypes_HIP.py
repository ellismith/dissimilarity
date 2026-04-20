import h5py
import numpy as np
import pandas as pd
from scipy import stats

region = "HIP"
n_pcs = 30

cell_type_files = {
    'astrocytes': 'Res1_astrocytes_update.h5ad',
    'basket-cells': 'Res1_basket-cells_update.h5ad',
    'cerebellar-neurons': 'Res1_cerebellar-neurons_subset.h5ad',
    'ependymal-cells': 'Res1_ependymal-cells_new.h5ad',
    'GABAergic-neurons': 'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons': 'Res1_glutamatergic-neurons_update.h5ad',
    'medium-spiny-neurons': 'Res1_medium-spiny-neurons_subset.h5ad',
    'microglia': 'Res1_microglia_new.h5ad',
    'midbrain-neurons': 'Res1_midbrain-neurons_update.h5ad',
    'opc-olig': 'Res1_opc-olig_subset.h5ad',
    'vascular-cells': 'Res1_vascular-cells_subset.h5ad',
}

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

results = []

for cell_type, fname in cell_type_files.items():
    fpath = f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{fname}'
    print(f"\nProcessing {cell_type}...")

    try:
        with h5py.File(fpath, 'r') as f:
            # check region col exists
            if 'region' not in f['obs']:
                print(f"  No region column, skipping")
                continue
            if 'X_pca' not in f['obsm']:
                print(f"  No X_pca, skipping")
                continue

            X_pca = f['obsm']['X_pca'][:, :n_pcs]
            regions = decode_categorical(f['obs']['region'])
            louvain = decode_categorical(f['obs']['louvain'])
            animal_ids = decode_categorical(f['obs']['animal_id'])
            ages = f['obs']['age'][:]

    except Exception as e:
        print(f"  Error loading: {e}")
        continue

    region_mask = (regions == region) & (ages >= 1)
    if region_mask.sum() == 0:
        print(f"  No cells in {region}, skipping")
        continue

    print(f"  {region_mask.sum()} cells in {region}")
    louvain_clusters = np.unique(louvain[region_mask])

    for cl in louvain_clusters:
        mask = region_mask & (louvain == cl)
        if mask.sum() < 10:
            continue

        pca_subset = X_pca[mask]
        centroid = pca_subset.mean(axis=0)
        diffs = pca_subset - centroid
        distances = np.sqrt((diffs ** 2).sum(axis=1))

        df = pd.DataFrame({
            'animal_id': animal_ids[mask],
            'age': ages[mask],
            'dist': distances
        })

        agg = df.groupby(['animal_id', 'age'])['dist'].mean().reset_index()

        if len(agg) < 5:
            continue

        slope, intercept, r, p, se = stats.linregress(agg['age'], agg['dist'])
        results.append({
            'cell_type': cell_type,
            'louvain': cl,
            'n_cells': mask.sum(),
            'n_animals': len(agg),
            'r': round(r, 3),
            'p': round(p, 4),
            'slope': round(slope, 5)
        })
        print(f"  louvain {cl}: n_cells={mask.sum()}, n_animals={len(agg)}, r={r:.2f}, p={p:.3f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p')

# FDR correction
from scipy.stats import rankdata
n = len(results_df)
ranks = rankdata(results_df['p'])
results_df['p_fdr'] = (results_df['p'] * n / ranks).clip(upper=1.0).round(4)

out = f'/scratch/easmit31/factor_analysis/centroid_dist_age_{region}_all_celltypes.csv'
results_df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(f"\nTop hits (p < 0.05):")
print(results_df[results_df['p'] < 0.05].to_string())
