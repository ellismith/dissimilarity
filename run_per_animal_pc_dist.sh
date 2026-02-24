#!/bin/bash
#SBATCH --job-name=pc_centroid
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/pc_centroid_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/pc_centroid_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4

mkdir -p /scratch/easmit31/factor_analysis/logs
mkdir -p /scratch/easmit31/factor_analysis/pc_centroid_outputs

CELL_TYPES=(
    "GABAergic-neurons"
    "glutamatergic-neurons"
    "astrocytes"
    "microglia"
    "opc"
    "oligodendrocytes"
    "vascular-cells"
    "ependymal-cells"
)

CELL_TYPE=${CELL_TYPES[$SLURM_ARRAY_TASK_ID]}
echo "Running cell type: $CELL_TYPE"

source activate latent_analysis

python /scratch/easmit31/factor_analysis/pca_centroid_all_regions.py \
    --cell_type $CELL_TYPE \
     \
    --n_pcs 50 \
    --min_cells 100 \
    --min_age 1.0 \
    --min_cells_per_animal 100 \
    --outdir /scratch/easmit31/factor_analysis/pc_centroid_outputs_min100/
