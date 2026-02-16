# Single-Cell Transcriptional Dissimilarity Analysis

Analysis pipeline for quantifying cell-level transcriptional similarity in aging primate brain single-cell RNA-seq data.

## Overview

This pipeline computes pairwise transcriptional distances between cells and uses k-nearest neighbor analysis to detect:
- Animal/batch effects
- Age-related clustering patterns
- Cell-type specific aging signatures

Key features:
- Multiple distance metrics (Euclidean, Correlation, Cosine)
- Raw vs z-scored expression comparison
- lochNESS-inspired age enrichment scoring
- Comprehensive validation framework

## Pipeline Components

### 1. Dissimilarity Matrix Computation
- `compute_dissimilarity_matrix.py` - Compute pairwise distances (raw expression)
- `compute_dissimilarity_matrix_zscore.py` - Compute pairwise distances (z-scored)

### 2. Analysis & Validation
- `analyze_dissimilarity_matrix.py` - K-nearest neighbor analysis
- `analyze_dissimilarity_matrix_no_animal_filter.py` - KNN without animal filter
- `validate_distances.py` - Four-part validation framework
- `validate_distances_no_animal_filter.py` - Validation without animal filter

### 3. Age Enrichment Analysis
- `compute_lochness_scores.py` - lochNESS-inspired age enrichment in neighborhoods

### 4. Comparison Tools
- `compare_all_distance_metrics.py` - Compare Euclidean/Correlation/Cosine
- `compare_raw_vs_zscore.py` - Compare raw vs z-scored results
- `visualize_raw_vs_zscore.py` - Comprehensive visualization
- `summarize_validation_results.py` - Aggregate results across analyses
- `summarize_lochness_results.py` - Aggregate lochNESS results
- `compare_subtype_enrichment.py` - Compare age enrichment across subtypes

### 5. Testing
- `test_dissimilarity_synthetic.py` - Validate methods with synthetic data

### 6. Batch Processing
- `batch_gabaergic_HIP.sh` - Process GABAergic neurons (hippocampus)
- `batch_gabaergic_dlPFC.sh` - Process GABAergic neurons (dlPFC)
- `batch_lochness_HIP.sh` - lochNESS analysis for hippocampus

## Quick Start

### Compute dissimilarity matrix for a cell type/region combination:
```bash
python compute_dissimilarity_matrix.py \
    --louvain 7 \
    --region EC \
    --h5ad /path/to/Res1_astrocytes_update.h5ad \
    --min-age 1
```

### Run KNN analysis and validation:
```bash
python analyze_dissimilarity_matrix.py \
    --dist-matrix output/louvain7_EC_minage1.0_distance_matrix.npy \
    --metadata output/louvain7_EC_minage1.0_cell_metadata.csv \
    --output-dir output/ \
    --k 10

python validate_distances.py \
    --dist-matrix output/louvain7_EC_minage1.0_distance_matrix.npy \
    --metadata output/louvain7_EC_minage1.0_cell_metadata.csv \
    --k 10
```

### Compare distance metrics:
```bash
python compare_all_distance_metrics.py \
    --louvain 7 \
    --region EC \
    --h5ad /path/to/Res1_astrocytes_update.h5ad
```

## Key Findings

- **Euclidean distance on raw expression** provides optimal structure (2.5x separation)
- **Z-scoring reduces biological signal** (1.4x separation)
- **Animal effects minimal** across all cell types (<2% difference)
- **Age effects subtle** but detectable (~1 year NN age difference)
- **Cell-type specific patterns**: GABAergic neurons show young-enrichment asymmetry

## Requirements
```
python >= 3.8
scanpy
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
```

## Citation

If you use this pipeline, please cite: [Your paper when published]

## License

MIT License
