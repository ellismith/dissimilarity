#!/bin/bash
#SBATCH --job-name=gaba_HIP_dissim
#SBATCH --output=/scratch/easmit31/dissimilarity_analysis/logs/gaba_HIP_%A_%a.out
#SBATCH --error=/scratch/easmit31/dissimilarity_analysis/logs/gaba_HIP_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15

# Create logs directory
mkdir -p /scratch/easmit31/dissimilarity_analysis/logs

# Activate conda environment
source ~/.bashrc
conda activate latent_analysis

# Paths
H5AD_FILE="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad"
OUTPUT_DIR="/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices"
SCRIPT_DIR="/scratch/easmit31/dissimilarity_analysis"

# HIP-only combinations (16 total)
COMBINATIONS=(
    "0:HIP"
    "1:HIP"
    "10:HIP"
    "11:HIP"
    "13:HIP"
    "14:HIP"
    "15:HIP"
    "16:HIP"
    "17:HIP"
    "18:HIP"
    "19:HIP"
    "2:HIP"
    "20:HIP"
    "21:HIP"
    "22:HIP"
    "23:HIP"
    "8:HIP"
    "9:HIP"
)

# Get the combination for this array task
COMBO=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
LOUVAIN=$(echo $COMBO | cut -d':' -f1)
REGION=$(echo $COMBO | cut -d':' -f2)

echo "=========================================="
echo "Processing GABAergic neurons - HIP only"
echo "Louvain: $LOUVAIN"
echo "Region: $REGION"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

# Step 1: Compute dissimilarity matrix
echo ""
echo "STEP 1: Computing dissimilarity matrix..."
python ${SCRIPT_DIR}/compute_dissimilarity_matrix.py \
    --louvain $LOUVAIN \
    --region $REGION \
    --h5ad $H5AD_FILE \
    --output-dir $OUTPUT_DIR \
    --min-age 1

# Check if distance matrix was created
DIST_MATRIX="${OUTPUT_DIR}/GABAergic-neurons/louvain${LOUVAIN}_${REGION}_minage1.0_distance_matrix.npy"
METADATA="${OUTPUT_DIR}/GABAergic-neurons/louvain${LOUVAIN}_${REGION}_minage1.0_cell_metadata.csv"

if [ ! -f "$DIST_MATRIX" ]; then
    echo "ERROR: Distance matrix not created. Exiting."
    exit 1
fi

# Step 2: Analyze with KNN
echo ""
echo "STEP 2: Performing KNN analysis..."
python ${SCRIPT_DIR}/analyze_dissimilarity_matrix.py \
    --dist-matrix $DIST_MATRIX \
    --metadata $METADATA \
    --output-dir ${OUTPUT_DIR}/GABAergic-neurons \
    --k 10

# Step 3: Validate distances
echo ""
echo "STEP 3: Validating distances..."
python ${SCRIPT_DIR}/validate_distances.py \
    --dist-matrix $DIST_MATRIX \
    --metadata $METADATA \
    --k 10

echo ""
echo "=========================================="
echo "✓ Analysis complete for Louvain $LOUVAIN, Region $REGION"
echo "=========================================="
