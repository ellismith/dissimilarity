import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

cell_class_file = "Res1_GABAergic-neurons_subset.h5ad"
region = "HIP"
n_pcs = 30

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

with h5py.File(f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{cell_class_file}', 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    barcodes = decode_col(f['obs']['_index'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages = f['obs']['age'][:]

region_mask = (regions == region) & (ages >= 1)
louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {region}")

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
        'dist_to_centroid': distances
    })

    agg = df.groupby(['animal_id', 'age'])['dist_to_centroid'].mean().reset_index()
    agg = agg.sort_values('age')

    slope, intercept, r, p, se = stats.linregress(agg['age'], agg['dist_to_centroid'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(agg['age'], agg['dist_to_centroid'], color='steelblue', s=40)
    x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='firebrick', lw=1.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Mean dist to centroid')
    ax.set_title(f'GABAergic {region} louvain {cl}\nr={r:.2f}, p={p:.3f}, n={len(agg)} animals')
    plt.tight_layout()

    out = f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{cl}_centroid_dist.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved louvain {cl} (n cells={mask.sum()}, n animals={len(agg)}, p={p:.3f})")

print("Done.")
