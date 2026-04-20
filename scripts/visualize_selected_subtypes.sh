#!/bin/bash

OUTPUT_DIR="/scratch/easmit31/factor_analysis/subtype_variability/figures"

echo "Visualizing PC scores for selected subtypes..."

# 1. Glutamatergic neurons_9 in HIP
python plot_pseudobulk_with_means.py \
    --cell_type Glutamatergic \
    --region HIP \
    --subtype "glutamatergic neurons_9" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Glutamatergic_HIP.csv \
    --output_dir $OUTPUT_DIR

# 2. Glutamatergic neurons_29 in IPP
python plot_pseudobulk_with_means.py \
    --cell_type Glutamatergic \
    --region IPP \
    --subtype "glutamatergic neurons_29" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Glutamatergic_IPP.csv \
    --output_dir $OUTPUT_DIR

# 3. Glutamatergic neurons_17 in M1
python plot_pseudobulk_with_means.py \
    --cell_type Glutamatergic \
    --region M1 \
    --subtype "glutamatergic neurons_17" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Glutamatergic_M1.csv \
    --output_dir $OUTPUT_DIR

# 4. GABAergic neurons_25 in EC
python plot_pseudobulk_with_means.py \
    --cell_type GABAergic \
    --region EC \
    --subtype "GABAergic neurons_25" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_GABAergic_EC.csv \
    --output_dir $OUTPUT_DIR

# 5. GABAergic neurons_18 in IPP
python plot_pseudobulk_with_means.py \
    --cell_type GABAergic \
    --region IPP \
    --subtype "GABAergic neurons_18" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_GABAergic_IPP.csv \
    --output_dir $OUTPUT_DIR

# 6. GABAergic neurons_11 in M1
python plot_pseudobulk_with_means.py \
    --cell_type GABAergic \
    --region M1 \
    --subtype "GABAergic neurons_11" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_GABAergic_M1.csv \
    --output_dir $OUTPUT_DIR

# 7. GABAergic neurons_12 in mdTN
python plot_pseudobulk_with_means.py \
    --cell_type GABAergic \
    --region mdTN \
    --subtype "GABAergic neurons_12" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_GABAergic_mdTN.csv \
    --output_dir $OUTPUT_DIR

# 8. Astrocytes_8 in HIP
python plot_pseudobulk_with_means.py \
    --cell_type Astrocytes \
    --region HIP \
    --subtype "astrocytes_8" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Astrocytes_HIP.csv \
    --output_dir $OUTPUT_DIR

# 9. Astrocytes_2 in IPP
python plot_pseudobulk_with_means.py \
    --cell_type Astrocytes \
    --region IPP \
    --subtype "astrocytes_2" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Astrocytes_IPP.csv \
    --output_dir $OUTPUT_DIR

# 10. Astrocytes_11 in mdTN
python plot_pseudobulk_with_means.py \
    --cell_type Astrocytes \
    --region mdTN \
    --subtype "astrocytes_11" \
    --h5ad_path /scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad \
    --pc_scores_path /scratch/easmit31/factor_analysis/csv_files/pca_analysis_Astrocytes_mdTN.csv \
    --output_dir $OUTPUT_DIR

echo "Done!"
