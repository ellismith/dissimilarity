# Dissimilarity Analysis Workflow

## Overview
Pipeline to analyze transcriptional similarity between single cells using pairwise distance matrices, k-nearest neighbor analysis, and lochNESS-inspired age enrichment scoring. Key principle: Same-animal cells ARE ALLOWED as neighbors to avoid assumptions about batch effects.

## Key Moving Parts

### 1. CORE ANALYSIS SCRIPTS

compute_dissimilarity_matrix.py
- Computes pairwise Euclidean distances between cells
- Filters by louvain cluster, region, and age
- Filters genes (keeps those expressed in >=5% of cells)
- Output: distance_matrix.npy + cell_metadata.csv

analyze_dissimilarity_matrix_no_animal_filter.py
- Finds k nearest neighbors for each cell (allows same-animal)
- Calculates distances and age differences to neighbors
- Counts same-animal vs different-animal neighbors
- Output: knn_analysis_no_animal_filter_k10.csv + plots

validate_distances_no_animal_filter.py
- Test 1: Same-animal vs different-animal distances
- Test 2: Nearest neighbor vs random distances (structure metric)
- Test 3: Same-animal composition of k-NN
- Test 4: Age differences in neighbors
- Output: validation_summary_no_animal_filter.csv + plots

compute_lochness_scores_no_animal_filter.py
- For each cell, calculates enrichment of old vs young neighbors
- lochNESS = (observed - expected) / expected
- Permutation testing for significance
- Output: lochness_scores_no_animal_filter.csv + plots

### 2. AUTOMATION SYSTEM

generate_batch_script.py
- Automatically generates SLURM batch scripts
- Finds all distance matrices for specified cell type and region
- Creates array job to process all combinations
- Usage: python generate_batch_script.py --cell-type GABAergic-neurons --region HIP --analysis both

process_one_combination.sh
- Wrapper script that runs KNN + validation for ONE distance matrix
- Called by batch jobs

process_one_lochness.sh
- Wrapper script that runs lochNESS for ONE distance matrix
- Separate from KNN because it's slower (permutation testing)

### 3. COMPARISON & SUMMARY TOOLS

compare_all_distance_metrics.py
- Compare Euclidean vs Correlation vs Cosine distances
- Test on raw vs z-scored expression
- Determine which captures most biological structure

summarize_validation_results.py
- Aggregate validation results across all louvain-region combinations
- Create summary tables and plots
- Usage: python summarize_validation_results.py --cell-type GABAergic-neurons

summarize_lochness_results.py
- Aggregate lochNESS results across all combinations
- Identify which subtypes show strongest age enrichment

compare_subtype_enrichment.py
- Compare young-enrichment vs old-enrichment across subtypes
- Identify subtypes with asymmetric aging patterns

## Typical Workflow

STEP 1: Generate batch script for your cell type and region
python generate_batch_script.py --cell-type GABAergic-neurons --region all --analysis both

STEP 2: Submit the generated batch script
sbatch batch_GABAergic-neurons_allregions_both.sh

STEP 3: Monitor progress
squeue -u easmit31
tail -f logs/GABA_all_*.out

STEP 4: After jobs complete, summarize results
python summarize_validation_results.py --cell-type GABAergic-neurons
python summarize_lochness_results.py --cell-type GABAergic-neurons --region HIP

## Key Design Decisions

Same-animal neighbors ALLOWED (no filter)
- More conservative approach
- Directly tests if cells cluster by animal ID
- Avoids assuming zero batch effects
- Results show only 6% same-animal neighbors (validates good QC)

Euclidean distance on raw expression
- Tested Euclidean vs Correlation vs Cosine
- Tested raw vs z-scored expression
- Euclidean on raw captures most structure (2.5x separation)
- Z-scoring removes biological signal (reduces to 1.4x)

K=10 nearest neighbors
- Based on lochNESS methodology (k = cube_root(N))
- Large enough to be robust, small enough to be local

Age threshold = 10 years for lochNESS
- Splits dataset roughly 50/50 young vs old
- Median age ~9 years in most datasets

## Output File Naming Convention

Distance matrices:
louvain{X}_{region}_minage{Y}_distance_matrix.npy
louvain{X}_{region}_minage{Y}_cell_metadata.csv

Analysis results:
louvain{X}_{region}_minage{Y}_knn_analysis_no_animal_filter_k10.csv
louvain{X}_{region}_minage{Y}_validation_summary_no_animal_filter.csv
louvain{X}_{region}_minage{Y}_lochness_scores_no_animal_filter.csv

All organized by cell type:
dissimilarity_matrices/{cell_type}/

## Quick Reference Commands

Generate batch script for GABAergic neurons, all regions, KNN only:
python generate_batch_script.py --cell-type GABAergic-neurons --region all --analysis knn

Generate batch script for astrocytes, HIP only, both KNN and lochNESS:
python generate_batch_script.py --cell-type astrocytes --region HIP --analysis both

Check job status:
squeue -u easmit31

Cancel a job:
scancel JOBID

Count completed files:
ls dissimilarity_matrices/GABAergic-neurons/*_knn_analysis_no_animal_filter_k10.csv | wc -l

Check for errors:
grep -i error logs/*.err

## Expected Results

Minimal animal clustering:
- 5-10% same-animal neighbors expected
- Indicates good batch correction

Strong biological structure:
- Random/NN ratio >2.0 = excellent
- Random/NN ratio 1.5-2.0 = good
- Random/NN ratio <1.5 = weak

Age effects:
- NN age difference ~1-2 years = weak age clustering
- Most cells show neutral lochNESS scores
- Small percentage (5-15%) show age enrichment

## Troubleshooting

Job fails with memory error:
- Increase --mem in batch script (default 32G)
- Very large combinations may need 64G

lochNESS taking too long:
- Reduce --n-permutations (default 100, can use 50 for testing)
- Run lochNESS separately from KNN (--analysis knn first, then --analysis lochness later)

Distance matrices already exist:
- Pipeline uses existing matrices, no need to recompute
- Just generate batch scripts for analysis steps

Wrong number of array tasks:
- generate_batch_script.py automatically counts matrices
- Check logs if some tasks exit immediately (may be missing metadata files)

