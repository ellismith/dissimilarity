#!/bin/bash
#SBATCH --job-name=lochness_HIP
#SBATCH --output=/scratch/easmit31/dissimilarity_analysis/logs/lochness_HIP_%A_%a.out
#SBATCH --error=/scratch/easmit31/dissimilarity_analysis/logs/lochness_HIP_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-15

# Create logs directory
mkdir -p /scratch/easmit31/dissimilarity_analysis/logs

# Activate conda environment
source ~/.bashrc
conda activate latent_analysis

# Paths
MATRIX_DIR="/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices/GABAergic-neurons"
SCRIPT_DIR="/scratch/easmit31/dissimilarity_analysis"

# HIP combinations (16 total)
LOUVAINS=(
    "0"
    "1"
    "2"
    "8"
    "9"
    "10"
    "11"
    "13"
    "14"
    "15"
    "16"
    "17"
    "18"
    "19"
    "20"
    "21"
    "22"
    "23"
)

# Get the louvain for this array task
LOUVAIN=${LOUVAINS[$SLURM_ARRAY_TASK_ID]}
REGION="HIP"

echo "=========================================="
echo "lochNESS Analysis: GABAergic HIP"
echo "Louvain: $LOUVAIN"
echo "Region: $REGION"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

# File paths
DIST_MATRIX="${MATRIX_DIR}/louvain${LOUVAIN}_${REGION}_minage1.0_distance_matrix.npy"
METADATA="${MATRIX_DIR}/louvain${LOUVAIN}_${REGION}_minage1.0_cell_metadata.csv"

# Check if files exist
if [ ! -f "$DIST_MATRIX" ]; then
    echo "ERROR: Distance matrix not found: $DIST_MATRIX"
    exit 1
fi

if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata not found: $METADATA"
    exit 1
fi

# Run lochNESS analysis
echo ""
echo "Running lochNESS analysis..."
python ${SCRIPT_DIR}/compute_lochness_scores.py \
    --dist-matrix $DIST_MATRIX \
    --metadata $METADATA \
    --k 10 \
    --age-threshold 10 \
    --n-permutations 100

echo ""
echo "=========================================="
echo "✓ lochNESS analysis complete for Louvain $LOUVAIN"
echo "=========================================="
