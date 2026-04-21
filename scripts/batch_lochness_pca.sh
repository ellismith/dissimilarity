#!/bin/bash
#SBATCH --job-name=lochness_pca
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/lochness_pca_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/lochness_pca_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-120

mkdir -p /scratch/easmit31/factor_analysis/logs

source ~/.bashrc
conda activate latent_analysis

CELL_TYPES=(
    "GABAergic-neurons"
    "glutamatergic-neurons"
    "astrocytes"
    "microglia"
    "basket-cells"
    "medium-spiny-neurons"
    "cerebellar-neurons"
    "ependymal-cells"
    "midbrain-neurons"
    "opc-olig"
    "vascular-cells"
)

REGIONS=("ACC" "CN" "dlPFC" "EC" "HIP" "IPP" "lCb" "M1" "MB" "mdTN" "NAc")

N_REGIONS=${#REGIONS[@]}

CELL_TYPE=${CELL_TYPES[$((SLURM_ARRAY_TASK_ID / N_REGIONS))]}
REGION=${REGIONS[$((SLURM_ARRAY_TASK_ID % N_REGIONS))]}

echo "=========================================="
echo "lochNESS (PCA)"
echo "Cell type: $CELL_TYPE"
echo "Region:    $REGION"
echo "=========================================="

python /scratch/easmit31/factor_analysis/compute_lochness_pca.py \
    --cell-type $CELL_TYPE \
    --region $REGION

echo "Done: $CELL_TYPE x $REGION"
