#!/bin/bash

# Run metric-k stability analysis for all GABAergic HIP louvains

LOUVAINS=(0 1 2 8 9 10 11 13 14 15 16 17 18 19 20 21)
REGION="HIP"
H5AD="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad"
SCRIPT="/scratch/easmit31/dissimilarity_analysis/analyze_metric_k_stability.py"

for LOUVAIN in "${LOUVAINS[@]}"; do
    echo "========================================"
    echo "Processing Louvain $LOUVAIN, $REGION"
    echo "========================================"
    
    python $SCRIPT \
        --louvain $LOUVAIN \
        --region $REGION \
        --h5ad $H5AD \
        --n-cells 10 \
        --k-values 5 10 15 20 30
    
    echo ""
done

echo "✓ All louvains complete!"

# Create summary
echo "Creating summary across all louvains..."
python /scratch/easmit31/dissimilarity_analysis/summarize_metric_k_stability.py
