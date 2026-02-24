"""
per_animal_pc_distributions.py

For each louvain cluster in a given cell type and brain region, plots the
distribution of PCA values across individual animals, sorted by age and
colored by age. Each PC gets its own row. Useful for visually inspecting
whether PC distributions shift with age within clusters.

Also annotates each animal's mean and variance of distance to their own
per-animal centroid (in n_pcs-dimensional space).

Usage:
    python per_animal_pc_distributions.py \
        --cell_type GABAergic-neurons \
        --region HIP \
        --n_pcs 10 \
        --min_cells 100 \
        --min_age 1.0 \
        --outdir /scratch/easmit31/factor_analysis/

Cell type options (matches filename):
    GABAergic-neurons, glutamatergic-neurons, astrocytes,
    microglia, opc-olig, vascular-cells, ependymal-cells
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- File map ---
CELL_TYPE_FILES = {
    'GABAergic-neurons':      'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons':  'Res1_glutamatergic-neurons_update.h5ad',
    'astrocytes':             'Res1_astrocytes_update.h5ad',
    'microglia':              'Res1_microglia_new.h5ad',
    'opc-olig':               'Res1_opc-olig_subset.h5ad',
    'vascular-cells':         'Res1_vascular-cells_subset.h5ad',
    'ependymal-cells':        'Res1_ependymal-cells_new.h5ad',
}
DATA_DIR = '/data/CEM/smacklab/U01/'

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--cell_type', default='GABAergic-neurons', choices=CELL_TYPE_FILES.keys())
parser.add_argument('--region',    default='HIP')
parser.add_argument('--n_pcs',     type=int, default=10, help='Number of PCs to plot')
parser.add_argument('--min_cells', type=int, default=100, help='Min total cells per louvain')
parser.add_argument('--min_age',   type=float, default=1.0)
parser.add_argument('--min_cells_per_animal', type=int, default=2)
parser.add_argument('--outdir',    default='/scratch/easmit31/factor_analysis/')
args = parser.parse_args()

# --- Helpers ---
def decode_categorical(grp):
    """Decode h5py categorical (categories + codes) to numpy string array."""
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

# --- Load ---
fpath = DATA_DIR + CELL_TYPE_FILES[args.cell_type]
print(f"Loading {fpath}...")
with h5py.File(fpath, 'r') as f:
    X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]

region_mask = (regions == args.region) & (ages >= args.min_age)
louvain_clusters = sorted(np.unique(louvain[region_mask]), key=lambda x: int(x))
print(f"Found {len(louvain_clusters)} louvains in {args.region}")

# --- Loop over louvains ---
for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    if mask.sum() < args.min_cells:
        print(f"  Skipping louvain {cl} (n={mask.sum()} < {args.min_cells})")
        continue

    cell_indices = np.where(mask)[0]
    animals = np.unique(animal_ids[cell_indices])
    animal_ages = {a: ages[cell_indices[animal_ids[cell_indices] == a][0]] for a in animals}
    animals_sorted = sorted(animals, key=lambda a: animal_ages[a])

    # compute per-animal centroids and distances
    animal_stats = {}
    for animal in animals_sorted:
        idxs = np.where(mask & (animal_ids == animal))[0]
        if len(idxs) < args.min_cells_per_animal:
            continue
        local_pca  = X_pca[idxs]
        centroid   = local_pca.mean(axis=0)
        dists      = np.sqrt(((local_pca - centroid) ** 2).sum(axis=1))
        animal_stats[animal] = {
            'age':       animal_ages[animal],
            'mean_dist': dists.mean(),
            'var_dist':  dists.var(ddof=1),
            'n_cells':   len(idxs),
            'pca':       local_pca,
        }

    animals_valid = [a for a in animals_sorted if a in animal_stats]
    if len(animals_valid) < 5:
        continue

    age_vals = np.array([animal_stats[a]['age'] for a in animals_valid])
    norm = plt.Normalize(age_vals.min(), age_vals.max())
    cmap = cm.plasma

    fig, axes = plt.subplots(args.n_pcs, 1, figsize=(16, 2.5 * args.n_pcs))
    if args.n_pcs == 1:
        axes = [axes]

    fig.suptitle(
        f'{args.cell_type} | {args.region} | louvain {cl} '
        f'(n={mask.sum()} cells, {len(animals_valid)} animals)\n'
        f'Per-animal PC distributions — sorted & colored by age',
        fontsize=10, y=1.01
    )

    for pc_idx, ax in enumerate(axes):
        for x_pos, animal in enumerate(animals_valid):
            color = cmap(norm(animal_stats[animal]['age']))
            vals  = animal_stats[animal]['pca'][:, pc_idx]
            parts = ax.violinplot([vals], positions=[x_pos],
                                  showmedians=True, showextrema=False, widths=0.75)
            for p in parts['bodies']:
                p.set_facecolor(color)
                p.set_alpha(0.8)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1)

        ax.set_ylabel(f'PC{pc_idx + 1}', fontsize=8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlim(-1, len(animals_valid))
        ax.set_xticks([])

        # on last PC row, show age + mean_dist + var_dist per animal
        if pc_idx == args.n_pcs - 1:
            ax.set_xticks(range(len(animals_valid)))
            ax.set_xticklabels(
                [f"{a}, {animal_stats[a]['age']:.1f}" for a in animals_valid],
                fontsize=7, rotation=60, ha='right'
            )
            ax.set_xlabel('Animal ID, Age (years)', fontsize=8)

    plt.tight_layout()
    out = f'{args.outdir}{args.cell_type}_{args.region}_louvain{cl}_per_animal_PC_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved louvain {cl} ({len(animals_valid)} animals)")

print("Done.")
