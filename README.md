# Transcriptional Individuality Analysis
Pipeline for quantifying how transcriptional individuality changes with age in rhesus macaque brain single-nucleus RNA-seq data (~55 animals, 11 brain regions, 11 cell types).

## Biological question
Does aging push cells toward a shared transcriptional state (age-based neighborhood clustering), or increase within-animal transcriptional heterogeneity (centroid spread)? These two phenomena can dissociate — a subtype can show increased age clustering without increased spread, and vice versa.

## Approaches

### 1. PCA centroid distance
For each louvain × region, measures how far each cell sits from its animal's own centroid in PCA space. Per-animal mean and variance of distances are regressed against age.

Two variants:
- **Within-animal centroid** (`pca_centroid_distance_by_louvain.py`): centroid computed per animal — measures within-animal transcriptional heterogeneity
- **Population centroid** (`pca_population_centroid_distance.py`): centroid computed across all cells — measures how far each animal drifts from the population mean

### 2. lochNESS (continuous age)
For each cell, finds k=10 nearest neighbors in PCA space and computes mean age of those neighbors. Tests whether a cell's own age correlates with its neighbors' mean age via Pearson r, against a permuted null. Positive r = old cells neighbor other old cells (age-based transcriptional clustering).

Per-cell output: `neighbor_mean_age`, `neighbor_age_zscore`
Per-louvain output: `r_obs`, `perm_pval`

## Data
- h5ad files: `/data/CEM/smacklab/U01/Res1_{cell_type}.h5ad`
- PCA coordinates: `obsm['X_pca']` (50 PCs, from original preprocessing)
- PCA embeddings (intermediate): `/scratch/easmit31/factor_analysis/pca_embeddings/{cell_type}/louvain{X}_{region}_pca.npy`

## Pipeline

### Step 1 — Extract PCA embeddings
```bash
sbatch scripts/batch_extract_pca_embeddings.sh
# runs extract_pca_embeddings.py for all 11 cell types x 11 regions
# output: pca_embeddings/{cell_type}/louvain{X}_{region}_pca.npy + _metadata.csv
```

### Step 2 — Compute lochNESS scores
```bash
sbatch scripts/batch_lochness_pca.sh
# runs compute_lochness_pca.py for all cell types x regions
# output: lochness_pca/{cell_type}/louvain{X}_{region}_lochness_scores.csv + _analysis.png
```

### Step 3 — Summarize lochNESS
```bash
for cell_type in GABAergic-neurons glutamatergic-neurons astrocytes microglia basket-cells medium-spiny-neurons cerebellar-neurons ependymal-cells midbrain-neurons opc-olig vascular-cells; do
    for region in ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc; do
        if ls lochness_pca/${cell_type}/*_${region}_lochness_scores.csv 2>/dev/null | grep -q .; then
            python scripts/summarize_lochness_results.py --cell-type $cell_type --region $region
        fi
    done
done
# output: lochness_pca/{cell_type}/lochness_summary_{region}.csv + .png
```

### Step 4 — Visualize
```bash
python scripts/plot_lochness_heatmap.py         # cell type x region heatmap
python scripts/plot_centroid_vs_lochness.py     # centroid r vs lochNESS r scatter
python scripts/plot_centroid_heatmaps.py        # centroid distance heatmaps
python scripts/plot_embedding_summary.py        # PCA embedding QC heatmaps
```

## Output directories
| Directory | Contents |
|---|---|
| `pca_embeddings/` | PCA coordinate arrays per louvain × region |
| `lochness_pca/` | Per-cell lochNESS scores, QC plots, summaries |
| `lochness_pca_binned_young_old/` | Old binary age threshold results (archived) |
| `pc_centroid_outputs_min100/` | Within-animal centroid distance summary CSVs |
| `population_centroid_outputs/` | Population centroid distance summary CSVs |
| `centroid_heatmaps/` | Centroid distance heatmaps per cell type |
| `scripts/` | Active analysis scripts |
| `scripts/factor_analysis_scripts/` | Factor analysis pipeline scripts |
| `scripts/other/` | GO enrichment and misc scripts |
| `scripts/archive/` | Superseded scripts |
| `archive/` | Superseded data and old results |

## Key parameters
| Parameter | Value |
|---|---|
| PCA components | 50 |
| Min cells per louvain | 100 |
| Min age | 1.0 years |
| lochNESS k | 10 |
| Permutations | 100 |
| Age metric | continuous (Pearson r) |

## Requirements
```
python >= 3.10
h5py
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
```
