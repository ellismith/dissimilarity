#!/bin/bash
#SBATCH --job-name=extract_pca_opcolig
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/extract_pca_opcolig_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/extract_pca_opcolig_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-10
#SBATCH --partition=htc

mkdir -p /scratch/easmit31/factor_analysis/logs

source ~/.bashrc
conda activate latent_analysis

REGIONS=("ACC" "CN" "dlPFC" "EC" "HIP" "IPP" "lCb" "M1" "MB" "mdTN" "NAc")
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Extract PCA embeddings: opc-olig split"
echo "Region: $REGION"
echo "=========================================="

python /scratch/easmit31/factor_analysis/scripts/extract_pca_embeddings_opcolig.py \
    --region $REGION

echo "Done: $REGION"
