# Dissimilarity Analysis Workflow

## Overview
This workflow computes pairwise transcriptional dissimilarity matrices for single-cell RNA-seq data, then analyzes k-nearest neighbor (KNN) patterns to detect age-related clustering and animal batch effects.

## Directory Structure
```
/scratch/easmit31/dissimilarity_analysis/
├── Core Analysis Scripts:
│   ├── compute_dissimilarity_matrix.py
│   ├── validate_distances_no_animal_filter.py
│   └── compute_lochness_scores_no_animal_filter.py
├── Comparison & Exploration:
│   ├── compare_all_distance_metrics.py
│   ├── compare_filtered_vs_unfiltered_batch.py
│   ├── calculate_neighbor_overlap.py
│   ├── analyze_metric_k_stability.py
│   └── explore_cell_neighbors.py
├── Batch Processing:
│   ├── generate_batch_script.py
│   ├── process_one_lochness.sh
│   └── batch_*.sh
├── Summarization:
│   ├── summarize_validation_results.py
│   ├── summarize_lochness_results.py
│   ├── compare_subtype_enrichment.py
│   └── summarize_metric_k_stability.py
└── dissimilarity_matrices/
    └── {cell_type}/
        ├── louvain{X}_{region}_distance_matrix.npy
        ├── louvain{X}_{region}_cell_metadata.csv
        ├── *_old.* (archived)
        └── no_animal_filter/
            ├── *_validation_summary_no_animal_filter.csv
            ├── *_lochness_scores_no_animal_filter.csv
            └── *.png
```

## Analysis Pipeline

### Step 1: Compute Distance Matrices
```bash
python compute_dissimilarity_matrix.py \
    --louvain 1 \
    --region HIP \
    --h5ad /path/to/cell_type.h5ad \
    --min-age 1.0 \
    --output-dir ./dissimilarity_matrices/GABAergic-neurons
```

Outputs: distance_matrix.npy, cell_metadata.csv

### Step 2: Validation Analysis
```bash
python validate_distances_no_animal_filter.py \
    --dist-matrix louvain1_HIP_minage1.0_distance_matrix.npy \
    --metadata louvain1_HIP_minage1.0_cell_metadata.csv \
    --k 10
```

Outputs: validation_summary_no_animal_filter.csv, validation_no_animal_filter.png

Checks: animal clustering, distance structure, age clustering

### Step 3: Age Enrichment (lochNESS)
```bash
python compute_lochness_scores_no_animal_filter.py \
    --dist-matrix louvain1_HIP_minage1.0_distance_matrix.npy \
    --metadata louvain1_HIP_minage1.0_cell_metadata.csv \
    --k 10 \
    --age-threshold 10 \
    --n-permutations 100
```

Outputs: lochness_scores_no_animal_filter.csv, lochness_analysis_no_animal_filter.png

Tests: young-enriched vs old-enriched cells

## Batch Processing

### Auto-generate batch scripts
```bash
python generate_batch_script.py \
    --cell-type GABAergic-neurons \
    --region all \
    --analysis both

sbatch batch_GABAergic-neurons_allregions_both.sh
```

### Monitor progress
```bash
squeue -u $USER
tail -f logs/job_*.out
```

## Summarization

### Validation summary across all combinations
```bash
python summarize_validation_results.py --cell-type GABAergic-neurons
```

### lochNESS summary
```bash
python summarize_lochness_results.py \
    --cell-type GABAergic-neurons \
    --region HIP
```

### Compare enrichment patterns
```bash
python compare_subtype_enrichment.py \
    --cell-type GABAergic-neurons \
    --region HIP
```

## Distance Metric Comparison

### Compare Euclidean vs Cosine vs Correlation
```bash
python compare_all_distance_metrics.py \
    --louvain 1 \
    --region HIP \
    --h5ad /path/to/GABAergic-neurons.h5ad \
    --output metric_comparison.csv
```

### Analyze metric stability
```bash
python analyze_metric_k_stability.py \
    --louvain 1 \
    --region HIP \
    --h5ad /path/to/GABAergic-neurons.h5ad \
    --n-cells 10 \
    --k-values 5 10 15 20 30
```

### Calculate neighbor overlap
```bash
python calculate_neighbor_overlap.py \
    --louvain 1 \
    --region HIP \
    --h5ad /path/to/GABAergic-neurons.h5ad \
    --k 10
```

## Interactive Exploration

### Explore specific cell neighbors
```bash
python explore_cell_neighbors.py \
    --cell-type GABAergic-neurons \
    --louvain 1 \
    --region HIP \
    --cell-index 100 \
    --distance-type raw_euclidean \
    --k 10
```

Or use interactive mode:
```bash
python explore_cell_neighbors.py --interactive
```

## Key Design Decisions

### Same-Animal Neighbors: ALLOWED
- More conservative approach
- Validates batch correction (only 6-10% same-animal neighbors)
- Age differences are real biology, not confounded

### Distance Metric: Euclidean on Raw Expression
- Captures magnitude information
- Standard approach
- Good performance (2.0-2.5x structure ratio)
- Note: Cosine/Correlation perform slightly better (5-10% more structure) but have 90% neighbor overlap with each other and only 10-15% overlap with Euclidean

### K-Nearest Neighbors: k=10
- Based on lochNESS methodology
- Stable across k=5 to k=30 (50-70% neighbor overlap)

### Age Threshold: 10 years
- Splits dataset approximately 50/50 young vs old
- Adjustable based on dataset

## Key Findings

### Batch Effects
- Excellent batch correction: 5-10% same-animal neighbors (when allowed)
- Animal effects minimal across all metrics and k values

### Distance Metrics
- Euclidean vs Cosine: only 10-15% neighbor overlap (fundamentally different)
- Cosine vs Correlation: 90-95% neighbor overlap (nearly identical)
- Raw expression >> Z-scored expression (2.0-2.5x vs 1.1-1.4x structure)
- Population statistics robust across metrics despite different neighbors

### K-Value Stability
- k=5 vs k=10: ~65% overlap
- k=10 vs k=20: ~50% overlap
- Stability consistent across all metrics

### Age Effects
- Weak age clustering: 1-6 years to nearest neighbor
- Some subtypes show young-enrichment (lochNESS)
- Age effects are subtle but present

## File Naming Convention

Pattern: `louvain{X}_{region}_minage{Y}_{analysis}_no_animal_filter_{suffix}`

Examples:
- `louvain1_HIP_minage1.0_distance_matrix.npy`
- `louvain1_HIP_minage1.0_validation_summary_no_animal_filter.csv`
- `louvain1_HIP_minage1.0_lochness_scores_no_animal_filter.csv`

## Troubleshooting

### Memory issues
- Reduce number of genes (stricter expression filter)
- Process smaller louvain clusters first
- Use sparse matrices where possible

### Long compute times
- lochNESS permutation testing is slow (100 permutations)
- Consider reducing n_permutations for testing
- Use batch jobs for production runs

### Missing outputs
- Check logs/ directory for errors
- Verify input files exist
- Check that louvain-region combination has >=100 cells and >=30 animals
