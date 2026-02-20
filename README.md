# Single-Cell Transcriptional Dissimilarity Analysis

Analysis pipeline for quantifying transcriptional individuality in aging primate brain single-cell RNA-seq data, using PCA centroid distance and lochNESS scoring.

## Overview

Two complementary approaches are used to measure how transcriptional individuality changes with age, at the level of cell type × region × louvain cluster:

**PCA centroid distance**: measures how far each cell is from the mean transcriptional state of its louvain cluster in PCA space. Higher distance = more transcriptionally atypical.

**lochNESS**: measures within-donor neighborhood enrichment in the kNN graph. For each cell, computes the fraction of same-donor neighbors relative to chance expectation. Higher score = cells from the same animal are more transcriptionally self-similar than expected.

These two metrics dissociate — a cluster can show increased donor-specificity (lochNESS) without cells drifting from the cluster mean (centroid distance), and vice versa.

## Data

- Cell-class h5ad files: `/data/CEM/smacklab/U01/Res1_*.h5ad`
- Region h5ad files: `/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update/`
- PCA coordinates read directly from `obsm['X_pca']` (50 components, stored from original preprocessing)
- kNN graph read from `obsp['distances']` (k=30, built on first 30 PCs)
- Output directory: `/scratch/easmit31/factor_analysis/`

## Scripts

### PCA Centroid Distance
- `pca_centroid_distance.py` - Single louvain/region/cell type centroid distance
- `pca_centroid_distance_by_louvain.py` - All louvains for GABAergic HIP, scatter plots vs age
- `pca_space_all_louvains.py` - PC1 vs PC2 plots colored by distance and age
- `pca_space_all_celltypes_HIP.py` - PC space plots for all HIP cell types
- `centroid_dist_age_all_celltypes_HIP.py` - Centroid distance vs age scatter plots, all HIP cell types
- `pca_centroid_age_all_celltypes_HIP.py` - Summary stats (r, p) across all HIP cell types and louvains

### lochNESS
- `lochness_score.py` - lochNESS per louvain with age regression summary
- `lochness_umap_louvain.py` - UMAP colored by per-cell lochNESS score
- `lochness_age_scatter.py` - Max lochNESS vs age per animal, all GABAergic HIP louvains

### Comparison
- `compare_centroid_lochness.py` - Side-by-side r/p table and scatter plot of both metrics
- `centroid_lochness_trend_comparison.py` - Visualize consistent and discordant trends across louvains

## lochNESS Formula

For cell_n and donor_m:
```
lochNESS = (# neighbors from donor_m in kNNs of cell_n / k) / (# cells from donor_m / N)
```

Where k=30 neighbors and N=total cells in the cluster. Score of 1 = no enrichment, >1 = within-donor enrichment. Formula from Schmitz et al. 2023 (Science Advances, doi:10.1126/sciadv.adh1914).

## Requirements
```
python >= 3.8
h5py
numpy
pandas
scipy
matplotlib
scikit-learn
```
