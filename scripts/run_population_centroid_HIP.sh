#!/bin/bash
# run_population_centroid_HIP.sh
# Runs pca_population_centroid_distance.py for all cell types in HIP
# Usage: bash run_population_centroid_HIP.sh

SCRIPT="/scratch/easmit31/factor_analysis/pca_population_centroid_distance.py"
H5AD_DIR="/data/CEM/smacklab/U01"
OUT_DIR="/scratch/easmit31/factor_analysis/population_centroid_outputs"
REGION="HIP"

declare -A CELL_TYPES=(
    ["GABAergic"]="Res1_GABAergic-neurons_subset.h5ad"
    ["Glutamatergic"]="Res1_glutamatergic-neurons_subset.h5ad"
    ["Astrocytes"]="Res1_astrocytes_update.h5ad"
    ["Oligodendrocytes"]="Res1_oligodendrocytes_subset.h5ad"
    ["OPC"]="Res1_OPC_subset.h5ad"
    ["Microglia"]="Res1_microglia_subset.h5ad"
    ["Vascular"]="Res1_vascular-cells_subset.h5ad"
    ["Ependymal"]="Res1_ependymal-cells_subset.h5ad"
)

for LABEL in "${!CELL_TYPES[@]}"; do
    H5AD="${H5AD_DIR}/${CELL_TYPES[$LABEL]}"
    if [ ! -f "$H5AD" ]; then
        echo "WARNING: File not found, skipping ${LABEL}: ${H5AD}"
        continue
    fi
    echo "=========================================="
    echo "Running ${LABEL} ${REGION}..."
    echo "=========================================="
    python "$SCRIPT" \
        --h5ad "$H5AD" \
        --cell-type "$LABEL" \
        --region "$REGION" \
        --output-dir "$OUT_DIR"
    echo ""
done

echo "All done."
