#!/bin/bash

# Output directory
OUTPUT_DIR="/scratch/easmit31/factor_analysis/subtype_variability"
mkdir -p $OUTPUT_DIR

# Cell type paths
GLUT_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad"
GABA_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad"
OPC_OLIGO_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_opc-olig_subset.h5ad"
MICRO_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad"

# PC scores directory
PC_DIR="/scratch/easmit31/factor_analysis/csv_files"

# Regions
REGIONS=(ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc)

echo "======================================================================="
echo "RUNNING REMAINING SUBTYPE VARIABILITY ANALYSES (Age >= 1)"
echo "======================================================================="

# Glutamatergic neurons - starting from EC onward
echo ""
echo "Processing Glutamatergic neurons (from EC onward)..."
for region in EC HIP IPP lCb M1 MB mdTN NAc; do
    echo "  - $region"
    python analyze_subtype_variability.py \
        --cell_type Glutamatergic \
        --region $region \
        --h5ad_path $GLUT_PATH \
        --pc_scores_path ${PC_DIR}/pca_analysis_Glutamatergic_${region}.csv \
        --subtype_col ct_louvain \
        --output_dir $OUTPUT_DIR \
        --min_animals 10 \
        --min_age 1.0
done

# GABAergic neurons - all regions
echo ""
echo "Processing GABAergic neurons..."
for region in "${REGIONS[@]}"; do
    echo "  - $region"
    python analyze_subtype_variability.py \
        --cell_type GABAergic \
        --region $region \
        --h5ad_path $GABA_PATH \
        --pc_scores_path ${PC_DIR}/pca_analysis_GABAergic_${region}.csv \
        --subtype_col ct_louvain \
        --output_dir $OUTPUT_DIR \
        --min_animals 10 \
        --min_age 1.0
done

# OPC_Oligo - all regions
echo ""
echo "Processing OPC_Oligo..."
for region in "${REGIONS[@]}"; do
    echo "  - $region"
    python analyze_subtype_variability.py \
        --cell_type OPC_Oligo \
        --region $region \
        --h5ad_path $OPC_OLIGO_PATH \
        --pc_scores_path ${PC_DIR}/pca_analysis_OPC_Oligo_${region}.csv \
        --subtype_col ct_louvain \
        --output_dir $OUTPUT_DIR \
        --min_animals 10 \
        --min_age 1.0
done

# Microglia - all regions
echo ""
echo "Processing Microglia..."
for region in "${REGIONS[@]}"; do
    echo "  - $region"
    python analyze_subtype_variability.py \
        --cell_type Microglia \
        --region $region \
        --h5ad_path $MICRO_PATH \
        --pc_scores_path ${PC_DIR}/pca_analysis_Microglia_${region}.csv \
        --subtype_col ct_louvain \
        --output_dir $OUTPUT_DIR \
        --min_animals 10 \
        --min_age 1.0
done

echo ""
echo "======================================================================="
echo "DONE! Results saved to: $OUTPUT_DIR"
echo "======================================================================="
