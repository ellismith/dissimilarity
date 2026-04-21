#!/bin/bash
#SBATCH --job-name=extract_pca
#SBATCH --output=/scratch/easmit31/dissimilarity_analysis/logs/extract_pca_%A_%a.out
#SBATCH --error=/scratch/easmit31/dissimilarity_analysis/logs/extract_pca_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-120

mkdir -p /scratch/easmit31/dissimilarity_analysis/logs

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
echo "Extract PCA embeddings"
echo "Cell type: $CELL_TYPE"
echo "Region:    $REGION"
echo "=========================================="

python /scratch/easmit31/dissimilarity_analysis/extract_pca_embeddings.py \
    --cell-type $CELL_TYPE \
    --region $REGION

echo "Done: $CELL_TYPE x $REGION"
