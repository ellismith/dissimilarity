"""
plot_pc12_animal_centroids.py

Plots cells in PC1/PC2 space for a given cell type x region x louvain,
showing per-animal centroids colored by age connected to the population
centroid, with optional confidence ellipses.

Usage:
    # Single louvain
    python plot_pc12_animal_centroids.py \
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \
        --cell-type GABAergic-neurons --region HIP --louvain 14

    # Panel of multiple louvains
    python plot_pc12_animal_centroids.py \
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \
        --cell-type GABAergic-neurons --region HIP --louvains 14 23 21

    # One louvain across all regions
    python plot_pc12_animal_centroids.py \
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \
        --cell-type GABAergic-neurons --louvain 14 --all-regions
"""
import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad',              required=True)
parser.add_argument('--cell-type',         required=True)
parser.add_argument('--region',            default=None)
parser.add_argument('--louvain',           default=None)
parser.add_argument('--louvains',          nargs='+')
parser.add_argument('--all-regions',       action='store_true')
parser.add_argument('--cell-class-filter', default=None)
parser.add_argument('--output-dir',        default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--min-age',           type=float, default=1.0)
parser.add_argument('--n-pcs',             type=int,   default=50)
parser.add_argument('--no-ellipses',       action='store_true')
parser.add_argument('--max-bg-cells',      type=int,   default=3000)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2, height=np.sqrt(1 - pearson) * 2, **kwargs)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

def plot_panel(ax, pca_2d, ages_sub, aids_sub, title, show_ellipses=True, max_bg=3000):
    norm = plt.Normalize(vmin=ages_sub.min(), vmax=ages_sub.max())
    cmap = cm.get_cmap('viridis')
    bg   = np.random.choice(len(pca_2d), min(max_bg, len(pca_2d)), replace=False)
    ax.scatter(pca_2d[bg, 0], pca_2d[bg, 1],
               c=[cmap(norm(ages_sub[i])) for i in bg],
               s=3, alpha=0.3, linewidths=0, rasterized=True)
    pop_c = pca_2d.mean(axis=0)
    ax.scatter(*pop_c, c='black', s=200, marker='*', zorder=10, label='Population centroid')
    for animal in np.unique(aids_sub):
        am = aids_sub == animal
        if am.sum() < 2:
            continue
        apca  = pca_2d[am]
        color = cmap(norm(ages_sub[am][0]))
        ac    = apca.mean(axis=0)
        ax.plot([pop_c[0], ac[0]], [pop_c[1], ac[1]], color=color, alpha=0.5, lw=0.8)
        ax.scatter(*ac, c=[color], s=40, edgecolors='black', linewidths=0.4, zorder=6)
        if show_ellipses and am.sum() >= 3:
            confidence_ellipse(apca[:, 0], apca[:, 1], ax, n_std=1.5,
                               facecolor=(*color[:3], 0.05),
                               edgecolor=(*color[:3], 0.4),
                               linewidth=0.6, zorder=4)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(title, fontsize=9)
    return norm, cmap

print(f"Loading {args.h5ad}...")
with h5py.File(args.h5ad, 'r') as f:
    X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]
    if args.cell_class_filter:
        cell_class = decode_categorical(f['obs']['cell_class_assign'])
    else:
        cell_class = None

base_mask = ages >= args.min_age
if cell_class is not None:
    base_mask = base_mask & (cell_class == args.cell_class_filter)

if args.all_regions:
    assert args.louvain is not None
    plot_regions  = sorted(np.unique(regions[base_mask]))
    plot_louvains = [args.louvain] * len(plot_regions)
    fname = f'{args.cell_type}_louvain{args.louvain}_allregions_pc12.png'
elif args.louvains:
    assert args.region is not None
    plot_louvains = args.louvains
    plot_regions  = [args.region] * len(plot_louvains)
    fname = f'{args.cell_type}_{args.region}_louvains{"_".join(plot_louvains)}_pc12.png'
else:
    assert args.louvain is not None and args.region is not None
    plot_louvains = [args.louvain]
    plot_regions  = [args.region]
    fname = f'{args.cell_type}_{args.region}_louvain{args.louvain}_pc12.png'

ncols = min(4, len(plot_louvains))
nrows = int(np.ceil(len(plot_louvains) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
axes = np.array(axes).reshape(-1)

norm_last, cmap_last = None, None
for i, (lou, reg) in enumerate(zip(plot_louvains, plot_regions)):
    mask = base_mask & (regions == reg) & (louvain == lou)
    if mask.sum() < 10:
        axes[i].set_visible(False)
        continue
    title = f'{args.cell_type} {reg} louvain {lou}\n(n={mask.sum():,} cells)'
    norm_last, cmap_last = plot_panel(axes[i], X_pca[mask, :2], ages[mask], animal_ids[mask],
                                      title, show_ellipses=not args.no_ellipses,
                                      max_bg=args.max_bg_cells)

for j in range(len(plot_louvains), len(axes)):
    axes[j].set_visible(False)

if norm_last is not None:
    sm = cm.ScalarMappable(cmap=cmap_last, norm=norm_last)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:len(plot_louvains)].tolist(), label='Age (years)', shrink=0.6)

plt.suptitle(f'{args.cell_type} — PC1/PC2 per-animal centroids', fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(args.output_dir, fname)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("Done.")
