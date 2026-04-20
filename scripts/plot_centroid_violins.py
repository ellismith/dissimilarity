"""
plot_centroid_violins.py

Four violin plots showing per-cell distance distributions:
  1. Per-region, within-animal centroid distance
  2. Per-region, population centroid distance
  3. Per-cell-type, within-animal centroid distance
  4. Per-cell-type, population centroid distance

Each violin = all per-cell distances for that region/cell type,
split into young (age < 10) and old (age >= 10) animals.

Usage:
    python plot_centroid_violins.py \
        --louvain 14 \
        --cell-type GABAergic-neurons \
        --region HIP
"""
import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--louvain',      default='14')
parser.add_argument('--cell-type',    default='GABAergic-neurons')
parser.add_argument('--region',       default='HIP')
parser.add_argument('--h5ad-dir',     default='/data/CEM/smacklab/U01')
parser.add_argument('--output-dir',   default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--min-age',      type=float, default=1.0)
parser.add_argument('--age-threshold',type=float, default=10.0)
parser.add_argument('--n-pcs',        type=int,   default=50)
parser.add_argument('--max-cells',    type=int,   default=5000,
                    help='Max cells per violin (subsampled for speed)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# h5ad filename map
H5AD_MAP = {
    'GABAergic-neurons':      'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons':  'Res1_glutamatergic-neurons_update.h5ad',
    'astrocytes':             'Res1_astrocytes_update.h5ad',
    'microglia':              'Res1_microglia_new.h5ad',
    'vascular-cells':         'Res1_vascular-cells_subset.h5ad',
    'ependymal-cells':        'Res1_ependymal-cells_new.h5ad',
    'basket-cells':           'Res1_basket-cells_update.h5ad',
    'cerebellar-neurons':     'Res1_cerebellar-neurons_subset.h5ad',
    'medium-spiny-neurons':   'Res1_medium-spiny-neurons_subset.h5ad',
    'midbrain-neurons':       'Res1_midbrain-neurons_update.h5ad',
    'opc':                    'Res1_opc-olig_subset.h5ad',
    'oligodendrocytes':       'Res1_opc-olig_subset.h5ad',
}

CELL_CLASS_FILTER = {
    'opc':             'oligodendrocyte precursor cells',
    'oligodendrocytes':'oligodendrocytes',
}

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

def load_h5ad(h5ad_path, cell_class_filter=None, n_pcs=50):
    with h5py.File(h5ad_path, 'r') as f:
        X_pca      = f['obsm']['X_pca'][:, :n_pcs]
        regions    = decode_categorical(f['obs']['region'])
        louvain    = decode_categorical(f['obs']['louvain'])
        animal_ids = decode_categorical(f['obs']['animal_id'])
        ages       = f['obs']['age'][:]
        if cell_class_filter:
            cell_class = decode_categorical(f['obs']['cell_class_assign'])
        else:
            cell_class = None
    return X_pca, regions, louvain, animal_ids, ages, cell_class

def get_distances(X_pca, regions, louvain, animal_ids, ages, cell_class,
                  region, lou, min_age, age_threshold, max_cells,
                  cell_class_filter=None):
    """Return dict with 'within' and 'pop' distance arrays, labeled by age group."""
    mask = (regions == region) & (louvain == lou) & (ages >= min_age)
    if cell_class_filter is not None:
        mask = mask & (cell_class == cell_class_filter)
    if mask.sum() < 10:
        return None

    idxs     = np.where(mask)[0]
    pca_sub  = X_pca[idxs]
    ages_sub = ages[idxs]
    aids_sub = animal_ids[idxs]

    # Population centroid
    pop_centroid = pca_sub.mean(axis=0)

    rows = []
    for animal in np.unique(aids_sub):
        am = aids_sub == animal
        if am.sum() < 2:
            continue
        local_pca   = pca_sub[am]
        age_val     = ages_sub[am][0]
        age_group   = 'old' if age_val >= age_threshold else 'young'

        # Within-animal centroid
        w_centroid  = local_pca.mean(axis=0)
        w_dists     = np.sqrt(((local_pca - w_centroid) ** 2).sum(axis=1))

        # Population centroid
        p_dists     = np.sqrt(((local_pca - pop_centroid) ** 2).sum(axis=1))

        for wd, pd_ in zip(w_dists, p_dists):
            rows.append({'age': age_val, 'age_group': age_group,
                         'within_dist': wd, 'pop_dist': pd_})

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Subsample if too many cells
    if len(df) > max_cells:
        df = df.sample(max_cells, random_state=42)

    return df

def violin_plot(ax, data_dict, metric, title, xlabel):
    """data_dict: {label: df} where df has age_group and metric columns."""
    labels  = list(data_dict.keys())
    young   = [data_dict[l][metric][data_dict[l]['age_group'] == 'young'].values for l in labels]
    old     = [data_dict[l][metric][data_dict[l]['age_group'] == 'old'].values   for l in labels]

    positions = np.arange(len(labels))
    width     = 0.35

    for pos, y_data, o_data in zip(positions, young, old):
        if len(y_data) > 1:
            vp = ax.violinplot([y_data], positions=[pos - width/2], widths=width,
                               showmedians=True, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor('steelblue')
                pc.set_alpha(0.7)
            vp['cmedians'].set_color('steelblue')

        if len(o_data) > 1:
            vp = ax.violinplot([o_data], positions=[pos + width/2], widths=width,
                               showmedians=True, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor('firebrick')
                pc.set_alpha(0.7)
            vp['cmedians'].set_color('firebrick')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Distance (PCA space)')
    ax.set_title(title, fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, label=f'Young (< {args.age_threshold}y)'),
                       Patch(facecolor='firebrick',  alpha=0.7, label=f'Old (≥ {args.age_threshold}y)')],
              fontsize=8)

# =====================
# Plot 1 & 2: Per-region (fixed cell type, all regions, louvain 14)
# =====================
print(f"Loading {args.cell_type} for per-region plots...")
h5ad_path = os.path.join(args.h5ad_dir, H5AD_MAP[args.cell_type])
ccf       = CELL_CLASS_FILTER.get(args.cell_type)
X_pca, regions, louvain, animal_ids, ages, cell_class = load_h5ad(h5ad_path, ccf, args.n_pcs)

region_data = {}
for reg in REGION_ORDER:
    df = get_distances(X_pca, regions, louvain, animal_ids, ages, cell_class,
                       reg, args.louvain, args.min_age, args.age_threshold,
                       args.max_cells, ccf)
    if df is not None:
        region_data[reg] = df

print(f"  Found data for {len(region_data)} regions")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
violin_plot(axes[0], region_data, 'within_dist',
            f'{args.cell_type} louvain {args.louvain}\nWithin-Animal Centroid Distance by Region',
            'Region')
violin_plot(axes[1], region_data, 'pop_dist',
            f'{args.cell_type} louvain {args.louvain}\nPopulation Centroid Distance by Region',
            'Region')
plt.tight_layout()
out = os.path.join(args.output_dir, f'{args.cell_type}_louvain{args.louvain}_violin_by_region.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# =====================
# Plot 3 & 4: Per-cell-type (fixed region, all cell types, louvain 14)
# =====================
print(f"\nLoading all cell types for per-cell-type plots (region={args.region})...")
celltype_data = {}

for ct, fname in H5AD_MAP.items():
    h5ad_path = os.path.join(args.h5ad_dir, fname)
    if not os.path.exists(h5ad_path):
        print(f"  Skipping {ct}: file not found")
        continue
    ccf = CELL_CLASS_FILTER.get(ct)
    print(f"  Loading {ct}...")
    X_pca, regions, louvain, animal_ids, ages, cell_class = load_h5ad(h5ad_path, ccf, args.n_pcs)
    df = get_distances(X_pca, regions, louvain, animal_ids, ages, cell_class,
                       args.region, args.louvain, args.min_age, args.age_threshold,
                       args.max_cells, ccf)
    if df is not None:
        celltype_data[ct] = df
        print(f"    {len(df)} cells")
    else:
        print(f"    No data")

print(f"Found data for {len(celltype_data)} cell types")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
violin_plot(axes[0], celltype_data, 'within_dist',
            f'{args.region} louvain {args.louvain}\nWithin-Animal Centroid Distance by Cell Type',
            'Cell Type')
violin_plot(axes[1], celltype_data, 'pop_dist',
            f'{args.region} louvain {args.louvain}\nPopulation Centroid Distance by Cell Type',
            'Cell Type')
plt.tight_layout()
out = os.path.join(args.output_dir, f'{args.region}_louvain{args.louvain}_violin_by_celltype.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("Done.")
