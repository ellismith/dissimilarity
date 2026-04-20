cat > /scratch/easmit31/factor_analysis/scripts/pca_centroid_distance_by_louvain.py << 'EOF'
"""
pca_centroid_distance_by_louvain.py
Within-animal centroid distance analysis.
"""
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad',               required=True)
parser.add_argument('--cell-type',          required=True)
parser.add_argument('--region',             required=True)
parser.add_argument('--output-dir',         default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--n-pcs',              type=int,   default=50)
parser.add_argument('--min-cells',          type=int,   default=100)
parser.add_argument('--min-age',            type=float, default=1.0)
parser.add_argument('--min-animals',        type=int,   default=5)
parser.add_argument('--cell-class-filter',  default=None,
                    help='Filter cells by cell_class_assign value (e.g. "oligodendrocytes")')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

print(f"Loading {args.h5ad}...")
with h5py.File(args.h5ad, 'r') as f:
    X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]
    if args.cell_class_filter is not None:
        cell_class = decode_categorical(f['obs']['cell_class_assign'])
    else:
        cell_class = None

# Base mask: region + age + optional cell class filter
region_mask = (regions == args.region) & (ages >= args.min_age)
if cell_class is not None:
    region_mask = region_mask & (cell_class == args.cell_class_filter)
    print(f"Filtering to cell_class_assign == '{args.cell_class_filter}'")

louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {args.region} ({region_mask.sum()} cells)")

summary_rows = []

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    if mask.sum() < args.min_cells:
        continue

    idxs = np.where(mask)[0]
    rows = []
    for animal in np.unique(animal_ids[idxs]):
        am = animal_ids[idxs] == animal
        if am.sum() < 2:
            continue
        local_pca = X_pca[idxs][am]
        centroid  = local_pca.mean(axis=0)
        dists     = np.sqrt(((local_pca - centroid) ** 2).sum(axis=1))
        rows.append({'animal_id': animal, 'age': ages[idxs][am][0],
                     'n_cells': am.sum(), 'mean_dist': dists.mean(), 'var_dist': dists.var()})

    df = pd.DataFrame(rows)
    if df.empty or len(df) < args.min_animals:
        continue

    df     = df.sort_values('age')
    df_var = df.dropna(subset=['var_dist'])

    slope_m, intercept_m, r_m, p_m, _ = stats.linregress(df['age'], df['mean_dist'])
    slope_v, intercept_v, r_v, p_v, _ = stats.linregress(df_var['age'], df_var['var_dist'])

    summary_rows.append({'cell_type': args.cell_type, 'region': args.region, 'louvain': cl,
                         'n_cells': mask.sum(), 'n_animals': len(df),
                         'r_mean': round(r_m, 4), 'p_mean': round(p_m, 4),
                         'r_var':  round(r_v, 4), 'p_var':  round(p_v, 4)})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, slope, intercept, r, p, ylabel in [
        (axes[0], 'mean_dist', slope_m, intercept_m, r_m, p_m, 'Mean dist to animal centroid'),
        (axes[1], 'var_dist',  slope_v, intercept_v, r_v, p_v, 'Variance of dist to animal centroid'),
    ]:
        agg = df if col == 'mean_dist' else df_var
        ax.scatter(agg['age'], agg[col], color='steelblue', s=40)
        x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='firebrick', lw=1.5)
        ax.set_xlabel('Age'); ax.set_ylabel(ylabel)
        ax.set_title(f'{args.cell_type} {args.region} louvain {cl}\nr={r:.2f}, p={p:.3f}, n={len(agg)}')
    plt.tight_layout()
    out = os.path.join(args.output_dir, f'{args.cell_type}_{args.region}_louvain{cl}_animal_centroid_mean_var.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved louvain {cl} (n cells={mask.sum()}, n animals={len(df)}, mean p={p_m:.3f}, var p={p_v:.3f})")

if summary_rows:
    new_df   = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output_dir, f'{args.cell_type}_centroid_summary.csv')
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[~((existing['cell_type'] == args.cell_type) & (existing['region'] == args.region))]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to {csv_path}")
else:
    print("No louvains passed filters.")

print("Done.")
EOF

cat > /scratch/easmit31/factor_analysis/scripts/pca_population_centroid_distance.py << 'EOF'
"""
pca_population_centroid_distance.py
Population centroid distance analysis.
"""
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad',               required=True)
parser.add_argument('--cell-type',          required=True)
parser.add_argument('--region',             required=True)
parser.add_argument('--output-dir',         default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--n-pcs',              type=int,   default=50)
parser.add_argument('--min-cells',          type=int,   default=100)
parser.add_argument('--min-age',            type=float, default=1.0)
parser.add_argument('--min-animals',        type=int,   default=5)
parser.add_argument('--cell-class-filter',  default=None,
                    help='Filter cells by cell_class_assign value (e.g. "oligodendrocytes")')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

print(f"Loading {args.h5ad}...")
with h5py.File(args.h5ad, 'r') as f:
    X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]
    if args.cell_class_filter is not None:
        cell_class = decode_categorical(f['obs']['cell_class_assign'])
    else:
        cell_class = None

region_mask = (regions == args.region) & (ages >= args.min_age)
if cell_class is not None:
    region_mask = region_mask & (cell_class == args.cell_class_filter)
    print(f"Filtering to cell_class_assign == '{args.cell_class_filter}'")

louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {args.region} ({region_mask.sum()} cells)")

summary_rows = []

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    if mask.sum() < args.min_cells:
        print(f"  Skipping louvain {cl}: only {mask.sum()} cells")
        continue

    idxs         = np.where(mask)[0]
    pop_centroid = X_pca[idxs].mean(axis=0)

    rows = []
    for animal in np.unique(animal_ids[idxs]):
        am = animal_ids[idxs] == animal
        if am.sum() < 2:
            continue
        local_pca = X_pca[idxs][am]
        dists     = np.sqrt(((local_pca - pop_centroid) ** 2).sum(axis=1))
        rows.append({'animal_id': animal, 'age': ages[idxs][am][0],
                     'n_cells': am.sum(), 'mean_dist': dists.mean(), 'var_dist': dists.var()})

    df = pd.DataFrame(rows).sort_values('age')
    if len(df) < args.min_animals:
        print(f"  Skipping louvain {cl}: only {len(df)} animals")
        continue

    df_var = df.dropna(subset=['var_dist'])
    slope_m, intercept_m, r_m, p_m, _ = stats.linregress(df['age'], df['mean_dist'])
    slope_v, intercept_v, r_v, p_v, _ = stats.linregress(df_var['age'], df_var['var_dist'])

    summary_rows.append({'louvain': cl, 'n_cells': mask.sum(), 'n_animals': len(df),
                         'r_mean_dist': r_m, 'p_mean_dist': p_m,
                         'r_var_dist':  r_v, 'p_var_dist':  p_v})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, slope, intercept, r, p, ylabel in [
        (axes[0], 'mean_dist', slope_m, intercept_m, r_m, p_m, 'Mean dist to population centroid'),
        (axes[1], 'var_dist',  slope_v, intercept_v, r_v, p_v, 'Variance of dist to population centroid'),
    ]:
        agg = df if col == 'mean_dist' else df_var
        ax.scatter(agg['age'], agg[col], color='steelblue', s=40)
        x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='firebrick', lw=1.5)
        ax.set_xlabel('Age'); ax.set_ylabel(ylabel)
        ax.set_title(f'{args.cell_type} {args.region} louvain {cl}\nr={r:.2f}, p={p:.3f}, n={len(agg)}')
    plt.tight_layout()
    out = os.path.join(args.output_dir, f'{args.cell_type}_{args.region}_louvain{cl}_population_centroid_mean_var.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved louvain {cl} (n_cells={mask.sum()}, n_animals={len(df)}, mean_r={r_m:.3f} p={p_m:.3f})")

if summary_rows:
    summary_df = pd.DataFrame(summary_rows).sort_values('louvain')
    csv_out    = os.path.join(args.output_dir, f'{args.cell_type}_{args.region}_population_centroid_summary.csv')
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary CSV saved to {csv_out}")
else:
    print("No louvains passed filters.")

print("Done.")
EOF

cat > /scratch/easmit31/factor_analysis/scripts/run_centroid_analyses.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=centroid_analysis
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/centroid_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/centroid_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-142

mkdir -p /scratch/easmit31/factor_analysis/logs

SCRIPTS=/scratch/easmit31/factor_analysis/scripts
H5AD_DIR=/data/CEM/smacklab/U01
OUT_WITHIN=/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100
OUT_POP=/scratch/easmit31/factor_analysis/population_centroid_outputs

# cell type entries: "h5ad_filename:cell_type_label:cell_class_filter"
# cell_class_filter is optional — leave blank for no filter
declare -a CELLTYPES=(
    "Res1_GABAergic-neurons_subset.h5ad:GABAergic-neurons:"
    "Res1_glutamatergic-neurons_update.h5ad:glutamatergic-neurons:"
    "Res1_astrocytes_update.h5ad:astrocytes:"
    "Res1_microglia_new.h5ad:microglia:"
    "Res1_vascular-cells_subset.h5ad:vascular-cells:"
    "Res1_ependymal-cells_new.h5ad:ependymal-cells:"
    "Res1_basket-cells_update.h5ad:basket-cells:"
    "Res1_cerebellar-neurons_subset.h5ad:cerebellar-neurons:"
    "Res1_medium-spiny-neurons_subset.h5ad:medium-spiny-neurons:"
    "Res1_midbrain-neurons_update.h5ad:midbrain-neurons:"
    "Res1_opc-olig_subset.h5ad:opc:oligodendrocyte precursor cells"
    "Res1_opc-olig_subset.h5ad:oligodendrocytes:oligodendrocytes"
    "Res1_opc-olig_subset.h5ad:midbrain-neurons:"
)

declare -a REGIONS=(ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc)

N_CT=${#CELLTYPES[@]}      # 13
N_RG=${#REGIONS[@]}        # 11

CT_IDX=$(( SLURM_ARRAY_TASK_ID / N_RG ))
RG_IDX=$(( SLURM_ARRAY_TASK_ID % N_RG ))

IFS=':' read -r H5AD_FILE CELL_TYPE CELL_CLASS_FILTER <<< "${CELLTYPES[$CT_IDX]}"
REGION="${REGIONS[$RG_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CELL_TYPE} × ${REGION} (filter: '${CELL_CLASS_FILTER}')"

H5AD_PATH="${H5AD_DIR}/${H5AD_FILE}"
if [ ! -f "$H5AD_PATH" ]; then
    echo "ERROR: h5ad not found: $H5AD_PATH"
    exit 1
fi

# Build optional filter argument
FILTER_ARG=""
if [ -n "$CELL_CLASS_FILTER" ]; then
    FILTER_ARG="--cell-class-filter ${CELL_CLASS_FILTER}"
fi

# Part 1
echo "--- Part 1: within-animal centroid ---"
python ${SCRIPTS}/pca_centroid_distance_by_louvain.py \
    --h5ad       ${H5AD_PATH} \
    --cell-type  ${CELL_TYPE} \
    --region     ${REGION} \
    --output-dir ${OUT_WITHIN} \
    ${FILTER_ARG}

# Part 2
echo "--- Part 2: population centroid ---"
python ${SCRIPTS}/pca_population_centroid_distance.py \
    --h5ad       ${H5AD_PATH} \
    --cell-type  ${CELL_TYPE} \
    --region     ${REGION} \
    --output-dir ${OUT_POP} \
    ${FILTER_ARG}

echo "Done: ${CELL_TYPE} × ${REGION}"
EOF

chmod +x /scratch/easmit31/factor_analysis/scripts/run_centroid_analyses.sh
echo "All scripts written."
