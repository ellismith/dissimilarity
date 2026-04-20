#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import scanpy as sc

print("="*70)
print("INSPECTING PCA & FA LOADINGS STRUCTURE")
print("="*70)

# Load data
print("\nLoading data...")

all_data = []
cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
paths = {
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad'
}

gene_names = None
gene_symbols = None

for cell_type in cell_types:
    adata = sc.read_h5ad(paths[cell_type], backed='r')
    
    if gene_names is None:
        gene_names = adata.var_names.to_list()
        gene_symbols = adata.var['external_gene_name'].tolist()
    
    obs_df = adata.obs[['animal_id', 'region']].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    
    unique_combos = obs_df[['animal_id', 'region']].drop_duplicates()
    
    for idx, row in unique_combos.iterrows():
        mask = (obs_df['animal_id'] == row['animal_id']) & (obs_df['region'] == row['region'])
        cell_indices = np.where(mask)[0]
        if len(cell_indices) == 0:
            continue
        pseudobulk = np.array(adata.X[cell_indices, :].sum(axis=0)).flatten()
        all_data.append(pseudobulk)

data_matrix = np.vstack(all_data)

# Filter & process
min_samples = int(0.2 * len(all_data))
gene_counts = (data_matrix > 0).sum(axis=0)
genes_keep = gene_counts >= min_samples
data_filtered = data_matrix[:, genes_keep]
gene_names_filtered = [gene_names[i] for i in range(len(gene_names)) if genes_keep[i]]
gene_symbols_filtered = [gene_symbols[i] for i in range(len(gene_symbols)) if genes_keep[i]]

cpm = (data_filtered / data_filtered.sum(axis=1, keepdims=True)) * 1e6
data_log = np.log2(cpm + 1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_log)

print(f"\nData shape: {data_scaled.shape} (samples × genes)")

# ==================== PCA WITH 50 COMPONENTS ====================
print("\n" + "="*70)
print("PCA STRUCTURE (50 COMPONENTS)")
print("="*70)

pca = PCA(n_components=50, random_state=42)
pca_scores = pca.fit_transform(data_scaled)

print(f"\nPCA Outputs:")
print(f"  pca.components_ shape: {pca.components_.shape}")
print(f"    → (n_components × n_genes) = (50 × {len(gene_symbols_filtered)})")
print(f"    → Each ROW is a PC")
print(f"    → Each COLUMN is a gene")
print(f"    → Values = how much each gene contributes to each PC")

print(f"\n  Example: pca.components_[0, :] = PC1 loadings for ALL genes")
print(f"  Example: pca.components_[:, 0] = ALL PCs loadings for gene 1")

# Show top 10 genes by absolute loading
print(f"\n  PC1 top 10 genes by |loading|:")
pc1_loadings = pca.components_[0, :]
top_indices = np.argsort(np.abs(pc1_loadings))[::-1][:10]
for idx in top_indices:
    print(f"    {gene_symbols_filtered[idx]:<20}: {pc1_loadings[idx]:>8.4f}")

print(f"\n  PC1 loading distribution:")
print(f"    Min: {pca.components_[0, :].min():.4f}")
print(f"    Max: {pca.components_[0, :].max():.4f}")
print(f"    Mean: {pca.components_[0, :].mean():.6f}")
print(f"    Std: {pca.components_[0, :].std():.4f}")

print(f"\n  Variance explained by each PC (first 20):")
for i in range(20):
    print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%")
print(f"  ...")
print(f"  Total variance explained (PC1-50): {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ==================== FA ====================
print("\n" + "="*70)
print("FACTOR ANALYSIS STRUCTURE (50 FACTORS)")
print("="*70)

fa = FactorAnalysis(n_components=50, random_state=42, max_iter=1000)
fa_scores = fa.fit_transform(data_scaled)

print(f"\nFA Outputs:")
print(f"  fa.components_ shape: {fa.components_.shape}")
print(f"    → (n_components × n_genes) = (50 × {len(gene_symbols_filtered)})")
print(f"    → Each ROW is a Factor")
print(f"    → Each COLUMN is a gene")
print(f"    → Values = how much each gene loads on each factor")

print(f"\n  Example: fa.components_[0, :] = Factor1 loadings for ALL genes")
print(f"  Example: fa.components_[:, 0] = ALL Factors loadings for gene 1")

# Show top 10 genes by absolute loading
print(f"\n  Factor1 top 10 genes by |loading|:")
f1_loadings = fa.components_[0, :]
top_indices = np.argsort(np.abs(f1_loadings))[::-1][:10]
for idx in top_indices:
    print(f"    {gene_symbols_filtered[idx]:<20}: {f1_loadings[idx]:>8.4f}")

print(f"\n  Factor1 loading distribution:")
print(f"    Min: {fa.components_[0, :].min():.4f}")
print(f"    Max: {fa.components_[0, :].max():.4f}")
print(f"    Mean: {fa.components_[0, :].mean():.6f}")
print(f"    Std: {fa.components_[0, :].std():.4f}")

# ==================== COMPARISON ====================
print("\n" + "="*70)
print("PCA-50 vs FA-50 LOADINGS COMPARISON")
print("="*70)

print(f"\nKey Differences:")
print(f"\n1. SCALE:")
print(f"   PCA loadings: typically -0.05 to +0.05")
print(f"   FA loadings: typically -0.10 to +0.10")
print(f"   → FA loadings are often larger magnitude")

print(f"\n2. INTERPRETATION:")
print(f"   PCA: Loadings show contribution to variance")
print(f"        High |loading| = gene strongly affects this PC")
print(f"   FA:  Loadings show relationship to latent factor")
print(f"        High |loading| = gene strongly driven by this factor")

print(f"\n3. SPARSITY:")
pca_sparse = (np.abs(pca.components_[0, :]) < 0.001).sum()
fa_sparse = (np.abs(fa.components_[0, :]) < 0.001).sum()
print(f"   PC1 near-zero loadings: {pca_sparse}/{len(gene_symbols_filtered)} ({pca_sparse/len(gene_symbols_filtered)*100:.1f}%)")
print(f"   Factor1 near-zero loadings: {fa_sparse}/{len(gene_symbols_filtered)} ({fa_sparse/len(gene_symbols_filtered)*100:.1f}%)")

# Show a specific gene across all components
example_gene_idx = 100
example_gene = gene_symbols_filtered[example_gene_idx]

print(f"\n4. EXAMPLE: Gene '{example_gene}' across components:")
print(f"\n   PCA (first 10 PCs):")
for i in range(10):
    print(f"     PC{i+1}: {pca.components_[i, example_gene_idx]:>8.4f}")

print(f"\n   FA (first 10 factors):")
for i in range(10):
    print(f"     Factor{i+1}: {fa.components_[i, example_gene_idx]:>8.4f}")

# ==================== SAVE FULL MATRICES ====================
print("\n" + "="*70)
print("SAVING FULL LOADING MATRICES (WITH GENE SYMBOLS)")
print("="*70)

# PCA loadings with gene symbols (50 components now!)
pca_loadings_df = pd.DataFrame(
    pca.components_,
    index=[f'PC{i+1}' for i in range(50)],
    columns=gene_symbols_filtered
)
pca_loadings_df.to_csv('pca_all_loadings_with_symbols.csv')
print(f"\nSaved: pca_all_loadings_with_symbols.csv")
print(f"  Shape: {pca_loadings_df.shape} (50 PCs × genes)")

# FA loadings with gene symbols
fa_loadings_df = pd.DataFrame(
    fa.components_,
    index=[f'Factor{i+1}' for i in range(50)],
    columns=gene_symbols_filtered
)
fa_loadings_df.to_csv('fa_all_loadings_with_symbols.csv')
print(f"\nSaved: fa_all_loadings_with_symbols.csv")
print(f"  Shape: {fa_loadings_df.shape} (50 Factors × genes)")

print("\n" + "="*70)
print("Done! Now PCA and FA both have 50 dimensions.")
print("="*70)

