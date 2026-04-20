#!/bin/bash

OUTPUT_DIR="/scratch/easmit31/factor_analysis/subtype_variability/figures"
PC_DIR="/scratch/easmit31/factor_analysis/csv_files"

GLUT_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad"
GABA_PATH="/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad"

# Regions
REGIONS=(ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc)

echo "======================================================================="
echo "PLOTTING PSEUDOBULK PC SCORES FOR ALL SUBTYPES"
echo "======================================================================="

# Glutamatergic
echo ""
echo "Processing Glutamatergic neurons..."
for region in "${REGIONS[@]}"; do
    echo "  Region: $region"
    
    # Get unique subtypes from the variability results
    if [ -f "${OUTPUT_DIR}/../subtype_variability_Glutamatergic_${region}.csv" ]; then
        # Read subtypes line by line to preserve spaces
        while IFS= read -r subtype; do
            if [ -n "$subtype" ]; then
                echo "    Subtype: $subtype"
                python plot_pseudobulk_pc_scores.py \
                    --cell_type Glutamatergic \
                    --region $region \
                    --subtype "$subtype" \
                    --h5ad_path $GLUT_PATH \
                    --pc_scores_path ${PC_DIR}/pca_analysis_Glutamatergic_${region}.csv \
                    --output_dir $OUTPUT_DIR \
                    --min_age 1.0 2>&1 | grep -E "(Matched|ERROR|Saved)"
            fi
        done < <(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/../subtype_variability_Glutamatergic_${region}.csv')
for subtype in df['subtype'].unique():
    print(subtype)
" 2>/dev/null)
    fi
done

# GABAergic
echo ""
echo "Processing GABAergic neurons..."
for region in "${REGIONS[@]}"; do
    echo "  Region: $region"
    
    # Get unique subtypes from the variability results
    if [ -f "${OUTPUT_DIR}/../subtype_variability_GABAergic_${region}.csv" ]; then
        # Read subtypes line by line to preserve spaces
        while IFS= read -r subtype; do
            if [ -n "$subtype" ]; then
                echo "    Subtype: $subtype"
                python plot_pseudobulk_pc_scores.py \
                    --cell_type GABAergic \
                    --region $region \
                    --subtype "$subtype" \
                    --h5ad_path $GABA_PATH \
                    --pc_scores_path ${PC_DIR}/pca_analysis_GABAergic_${region}.csv \
                    --output_dir $OUTPUT_DIR \
                    --min_age 1.0 2>&1 | grep -E "(Matched|ERROR|Saved)"
            fi
        done < <(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/../subtype_variability_GABAergic_${region}.csv')
for subtype in df['subtype'].unique():
    print(subtype)
" 2>/dev/null)
    fi
done

echo ""
echo "======================================================================="
echo "DONE!"
echo "======================================================================="
