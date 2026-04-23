#!/bin/bash
#SBATCH --job-name=centroid_full
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/centroid_full_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/centroid_full_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-120
#SBATCH --partition=htc

mkdir -p /scratch/easmit31/factor_analysis/logs
source ~/.bashrc
conda activate latent_analysis

REGIONS=(ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc)

JOBS=()
for region in "${REGIONS[@]}"; do
    JOBS+=("GABAergic-neurons:Res1_GABAergic-neurons_subset.h5ad:${region}:")
    JOBS+=("glutamatergic-neurons:Res1_glutamatergic-neurons_update.h5ad:${region}:")
    JOBS+=("astrocytes:Res1_astrocytes_update.h5ad:${region}:")
    JOBS+=("microglia:Res1_microglia_new.h5ad:${region}:")
    JOBS+=("basket-cells:Res1_basket-cells_update.h5ad:${region}:")
    JOBS+=("medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:${region}:")
    JOBS+=("cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:${region}:")
    JOBS+=("ependymal-cells:Res1_ependymal-cells_new.h5ad:${region}:")
    JOBS+=("midbrain-neurons:Res1_midbrain-neurons_update.h5ad:${region}:")
    JOBS+=("OPCs:Res1_opc-olig_subset.h5ad:${region}:oligodendrocyte precursor cells")
    JOBS+=("oligodendrocytes:Res1_opc-olig_subset.h5ad:${region}:oligodendrocytes")
done

echo "Total jobs: ${#JOBS[@]}"

ENTRY="${JOBS[$SLURM_ARRAY_TASK_ID]}"
CELL_TYPE=$(echo "$ENTRY" | cut -d: -f1)
H5AD=$(echo "$ENTRY" | cut -d: -f2)
REGION=$(echo "$ENTRY" | cut -d: -f3)
FILTER=$(echo "$ENTRY" | cut -d: -f4)

echo "=========================================="
echo "Centroid: $CELL_TYPE x $REGION"
echo "=========================================="

if [ -n "$FILTER" ]; then
    FILTER_ARG="--cell-class-filter $FILTER"
else
    FILTER_ARG=""
fi

python /scratch/easmit31/factor_analysis/scripts/pca_centroid_distance_by_louvain.py \
    --h5ad /data/CEM/smacklab/U01/$H5AD \
    --cell-type "$CELL_TYPE" \
    --region "$REGION" \
    $FILTER_ARG

echo "Done: $CELL_TYPE x $REGION"
