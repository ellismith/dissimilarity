"""
plot_pca_population_centroid_2d.py

Visualizes cells in PC1/PC2 space for a given louvain × region, showing:
  - All cells colored by animal age
  - Population centroid (black star)
  - Per-animal centroids colored by age, connected to population centroid by lines
  - Optional ellipses around each animal's cells

Can plot a single louvain or a panel of the top N most significant louvains
from the population centroid summary CSV.

Usage:
    # Single louvain
    python plot_pca_population_centroid_2d.py \\
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \\
        --region HIP \\
        --louvain 14 \\
        --cell-type GABAergic

    # Panel of top significant louvains from summary CSV
    python plot_pca_population_centroid_2d.py \\
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \\
        --region HIP \\
        --cell-type GABAergic \\
        --summary-csv /scratch/easmit31/factor_analysis/population_centroid_outputs/GABAergic_HIP_population_centroid_summary.csv \\
        --top-n 6 \\
        --output-dir /scratch/easmit31/factor_analysis/population_centroid_outputs/
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# --- Helpers ---

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    """Draw a covariance ellipse around points x, y."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_one_louvain(ax, pca_subset, animal_ids_sub, ages_sub, louvain_id, region,
                     cell_type, r_mean, p_mean, min_age, show_ellipses=True,
                     age_cmap='viridis', max_cells_bg=3000):
    """Plot PC1/PC2 for one louvain on a given axes."""

    # Age colormap
    age_min, age_max = ages_sub.min(), ages_sub.max()
    norm = plt.Normalize(vmin=age_min, vmax=age_max)
    cmap = cm.get_cmap(age_cmap)

    # Subsample background cells if too many
    n_cells = len(pca_subset)
    if n_cells > max_cells_bg:
        idx = np.random.choice(n_cells, max_cells_bg, replace=False)
    else:
        idx = np.arange(n_cells)

    # Background cells
    cell_colors = [cmap(norm(ages_sub[i])) for i in idx]
    ax.scatter(pca_subset[idx, 0], pca_subset[idx, 1],
               c=cell_colors, s=3, alpha=0.3, linewidths=0, rasterized=True)

    # Population centroid
    pop_centroid = pca_subset.mean(axis=0)
    ax.scatter(pop_centroid[0], pop_centroid[1],
               c='black', s=200, marker='*', zorder=10, label='Population centroid')

    # Per-animal centroids + lines + optional ellipses
    animals = np.unique(animal_ids_sub)
    for animal in animals:
        amask = animal_ids_sub == animal
        if amask.sum() < 2:
            continue
        animal_pca = pca_subset[amask]
        animal_age = ages_sub[amask][0]
        color = cmap(norm(animal_age))
        centroid = animal_pca.mean(axis=0)

        # Line from population centroid to animal centroid
        ax.plot([pop_centroid[0], centroid[0]],
                [pop_centroid[1], centroid[1]],
                color=color, alpha=0.5, lw=0.8, zorder=5)

        # Animal centroid dot
        ax.scatter(centroid[0], centroid[1],
                   c=[color], s=40, edgecolors='black', linewidths=0.4,
                   zorder=6)

        # Ellipse around animal's cells
        if show_ellipses and amask.sum() >= 3:
            confidence_ellipse(animal_pca[:, 0], animal_pca[:, 1], ax,
                               n_std=1.5,
                               facecolor=(*color[:3], 0.05),
                               edgecolor=(*color[:3], 0.4),
                               linewidth=0.6, zorder=4)

    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    p_str = f'{p_mean:.3f}' if p_mean >= 0.001 else f'{p_mean:.2e}'
    ax.set_title(f'{cell_type} {region} louvain {louvain_id}\n'
                 f'r={r_mean:.2f}, p={p_str}', fontsize=9)
    ax.tick_params(labelsize=7)

    return norm, cmap


def load_data(h5ad_path, region, min_age=1.0, n_pcs=50):
    print(f"Loading {h5ad_path}...")
    with h5py.File(h5ad_path, 'r') as f:
        X_pca      = f['obsm']['X_pca'][:, :n_pcs]
        regions    = decode_categorical(f['obs']['region'])
        louvain    = decode_categorical(f['obs']['louvain'])
        animal_ids = decode_categorical(f['obs']['animal_id'])
        ages       = f['obs']['age'][:]
    region_mask = (regions == region) & (ages >= min_age)
    print(f"  {region_mask.sum()} cells in {region} (age >= {min_age})")
    return X_pca, regions, louvain, animal_ids, ages, region_mask


def make_single_plot(args, X_pca, louvain, animal_ids, ages, region_mask,
                     louvain_id, r_mean=None, p_mean=None):
    mask = region_mask & (louvain == str(louvain_id))
    if mask.sum() == 0:
        print(f"No cells found for louvain {louvain_id} × {args.region}")
        return

    idxs = np.where(mask)[0]
    pca_sub  = X_pca[idxs, :2]
    aids_sub = animal_ids[idxs]
    ages_sub = ages[idxs]

    if r_mean is None or p_mean is None:
        # Compute quickly from per-animal means
        from scipy import stats
        rows = []
        for a in np.unique(aids_sub):
            am = aids_sub == a
            if am.sum() < 2:
                continue
            pop_c = pca_sub.mean(axis=0)  # approximate with 2D
            dists = np.sqrt(((pca_sub[am] - pop_c) ** 2).sum(axis=1))
            rows.append({'age': ages_sub[am][0], 'mean_dist': dists.mean()})
        df = pd.DataFrame(rows)
        _, _, r_mean, p_mean, _ = stats.linregress(df['age'], df['mean_dist'])

    fig, ax = plt.subplots(figsize=(7, 6))
    norm, cmap = plot_one_louvain(ax, pca_sub, aids_sub, ages_sub,
                                  louvain_id, args.region, args.cell_type,
                                  r_mean, p_mean, args.min_age,
                                  show_ellipses=args.ellipses)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Age')

    plt.tight_layout()
    out = os.path.join(args.output_dir,
                       f'{args.cell_type}_{args.region}_louvain{louvain_id}_pc12_popcentroid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def make_panel_plot(args, X_pca, louvain, animal_ids, ages, region_mask, summary_df):
    # Pick top N by abs(r_mean_dist), filtered by p < 0.05
    sig = summary_df[summary_df['p_mean_dist'] < 0.05].copy()
    sig['abs_r'] = sig['r_mean_dist'].abs()
    top = sig.nlargest(args.top_n, 'abs_r')

    if len(top) == 0:
        print("No significant louvains found (p_mean_dist < 0.05). Using top N by |r| regardless.")
        summary_df['abs_r'] = summary_df['r_mean_dist'].abs()
        top = summary_df.nlargest(args.top_n, 'abs_r')

    n = len(top)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    norm_global = None
    cmap_global = None

    for i, (_, row) in enumerate(top.iterrows()):
        lid = str(row['louvain'])
        mask = region_mask & (louvain == lid)
        if mask.sum() == 0:
            axes[i].set_visible(False)
            continue
        idxs = np.where(mask)[0]
        pca_sub  = X_pca[idxs, :2]
        aids_sub = animal_ids[idxs]
        ages_sub = ages[idxs]

        norm_global, cmap_global = plot_one_louvain(
            axes[i], pca_sub, aids_sub, ages_sub,
            lid, args.region, args.cell_type,
            row['r_mean_dist'], row['p_mean_dist'], args.min_age,
            show_ellipses=args.ellipses
        )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared colorbar
    if norm_global is not None:
        sm = cm.ScalarMappable(cmap=cmap_global, norm=norm_global)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:n].tolist(), label='Age', shrink=0.6)

    fig.suptitle(f'{args.cell_type} {args.region} — Top {n} louvains by population centroid displacement',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(args.output_dir,
                       f'{args.cell_type}_{args.region}_top{n}_pc12_popcentroid_panel.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved panel: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad',        required=True)
    parser.add_argument('--region',      required=True)
    parser.add_argument('--cell-type',   required=True, help='Label for titles/filenames')
    parser.add_argument('--louvain',     default=None,  help='Single louvain ID')
    parser.add_argument('--summary-csv', default=None,  help='Population centroid summary CSV for panel mode')
    parser.add_argument('--top-n',       type=int, default=6, help='Top N louvains for panel mode')
    parser.add_argument('--min-age',     type=float, default=1.0)
    parser.add_argument('--n-pcs',       type=int,   default=50)
    parser.add_argument('--no-ellipses', dest='ellipses', action='store_false',
                        help='Disable per-animal ellipses')
    parser.add_argument('--output-dir',  default='/scratch/easmit31/factor_analysis/population_centroid_outputs/')
    parser.set_defaults(ellipses=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_pca, regions, louvain, animal_ids, ages, region_mask = load_data(
        args.h5ad, args.region, args.min_age, args.n_pcs)

    if args.louvain is not None:
        # Single louvain mode
        make_single_plot(args, X_pca, louvain, animal_ids, ages, region_mask, args.louvain)

    elif args.summary_csv is not None:
        # Panel mode from summary CSV
        summary_df = pd.read_csv(args.summary_csv)
        make_panel_plot(args, X_pca, louvain, animal_ids, ages, region_mask, summary_df)

    else:
        parser.error("Provide either --louvain or --summary-csv")

    print("Done.")


if __name__ == '__main__':
    main()
