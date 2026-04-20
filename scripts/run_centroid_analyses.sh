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
