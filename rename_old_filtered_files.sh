#!/bin/bash
# Rename old filtered files (without no_animal_filter) to have _old suffix

CELL_TYPES=("astrocytes" "GABAergic-neurons" "glutamatergic-neurons" "medium-spiny-neurons" "opc-olig" "vascular-cells")

for CELL_TYPE in "${CELL_TYPES[@]}"; do
    DIR="/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices/${CELL_TYPE}"
    
    if [ ! -d "$DIR" ]; then
        echo "Skipping ${CELL_TYPE} (directory not found)"
        continue
    fi
    
    echo "Processing ${CELL_TYPE}..."
    
    # Find old filtered files (without no_animal_filter, without zscore, without _old already)
    # KNN files
    for f in ${DIR}/*_knn_analysis_k10.csv; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/_knn_analysis_k10.csv/_knn_analysis_k10_old.csv}"
            mv "$f" "$new"
            echo "  Renamed: $(basename $f) -> $(basename $new)"
        fi
    done
    
    for f in ${DIR}/*_knn_analysis_k10.png; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/_knn_analysis_k10.png/_knn_analysis_k10_old.png}"
            mv "$f" "$new"
        fi
    done
    
    # Validation files
    for f in ${DIR}/*_validation_summary.csv; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/_validation_summary.csv/_validation_summary_old.csv}"
            mv "$f" "$new"
            echo "  Renamed: $(basename $f) -> $(basename $new)"
        fi
    done
    
    for f in ${DIR}/*_validation.png; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]] && [[ ! "$f" =~ "zscore" ]]; then
            new="${f/_validation.png/_validation_old.png}"
            mv "$f" "$new"
        fi
    done
    
    # Heatmaps
    for f in ${DIR}/*_heatmap_*.png; do
        if [ -f "$f" ] && [[ ! "$f" =~ "zscore" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/.png/_old.png}"
            mv "$f" "$new"
            echo "  Renamed: $(basename $f) -> $(basename $new)"
        fi
    done
    
    # lochNESS files (old ones if they exist)
    for f in ${DIR}/*_lochness_scores.csv; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/_lochness_scores.csv/_lochness_scores_old.csv}"
            mv "$f" "$new"
            echo "  Renamed: $(basename $f) -> $(basename $new)"
        fi
    done
    
    for f in ${DIR}/*_lochness_analysis.png; do
        if [ -f "$f" ] && [[ ! "$f" =~ "no_animal_filter" ]] && [[ ! "$f" =~ "_old" ]]; then
            new="${f/_lochness_analysis.png/_lochness_analysis_old.png}"
            mv "$f" "$new"
        fi
    done
    
    echo ""
done

echo "✓ Done renaming old filtered files to *_old.*"
