#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats

print("="*70)
print("COMPARING PCA vs FACTOR ANALYSIS GENE LOADINGS")
print("="*70)

# Load summary files to know which components were age-associated
fa_summary = pd.read_csv('factor_analysis_all_celltypes_summary.csv')
pca_summary = pd.read_csv('pca_analysis_all_celltypes_summary.csv')

print(f"\nFA summary: {len(fa_summary)} cell type-region combos")
print(f"PCA summary: {len(pca_summary)} cell type-region combos")

# Merge summaries to match FA and PCA by cell type + region
matched = fa_summary.merge(
    pca_summary,
    on=['cell_type', 'region'],
    suffixes=('_fa', '_pca')
)

print(f"Matched: {len(matched)} cell type-region combos with both FA and PCA results\n")

# For each matched pair, compare gene loadings
all_results = []

for idx, row in matched.iterrows():
    celltype = row['cell_type']
    region = row['region']
    best_factor = row['best_factor']
    best_pc = row['best_pc']
    
    print(f"{'='*60}")
    print(f"{celltype} - {region}")
    print(f"  FA: {best_factor} (r={row['best_age_corr_fa']:.3f}, p={row['best_pval_fa']:.2e})")
    print(f"  PCA: {best_pc} (r={row['best_age_corr_pca']:.3f}, p={row['best_pval_pca']:.2e})")
    print(f"{'='*60}")
    
    # Use _with_symbols version for FA files
    fa_file = f'gene_loadings_{celltype}_{region}_{best_factor}_with_symbols.csv'
    pca_file = f'pca_gene_loadings_{celltype}_{region}_{best_pc}.csv'
    
    # Check if files exist
    try:
        fa_df = pd.read_csv(fa_file)
        print(f"Loaded FA genes: {len(fa_df)}")
    except FileNotFoundError:
        print(f"WARNING: FA file not found: {fa_file}")
        continue
    
    try:
        pca_df = pd.read_csv(pca_file)
        print(f"Loaded PCA genes: {len(pca_df)}")
    except FileNotFoundError:
        print(f"WARNING: PCA file not found: {pca_file}")
        continue
    
    # Get top N genes from each
    n_top = 100
    pca_top = set(pca_df.head(n_top)['gene_symbol'])
    fa_top = set(fa_df.head(n_top)['gene_symbol'])
    
    # Calculate overlap
    overlap = pca_top & fa_top
    n_overlap = len(overlap)
    overlap_pct = (n_overlap / n_top) * 100
    
    print(f"Top {n_top} gene overlap: {n_overlap} ({overlap_pct:.1f}%)")
    
    # Merge on gene symbol to get loadings for common genes
    merged = pca_df.merge(
        fa_df,
        left_on='gene_symbol',
        right_on='gene_symbol',
        suffixes=('_pca', '_fa')
    )
    
    if len(merged) > 10:  # Need reasonable number of genes
        # Correlate loadings (absolute values)
        corr_abs, pval_abs = stats.pearsonr(
            np.abs(merged['loading_pca']),
            np.abs(merged['loading_fa'])
        )
        print(f"Loading correlation (abs): r = {corr_abs:.3f}, p = {pval_abs:.3e}")
        
        # Also correlate raw loadings (checks sign agreement too)
        corr_raw, pval_raw = stats.pearsonr(
            merged['loading_pca'],
            merged['loading_fa']
        )
        print(f"Loading correlation (raw): r = {corr_raw:.3f}, p = {pval_raw:.3e}")
        
        # Sign concordance
        sign_concordance = (
            np.sign(merged['loading_pca']) == np.sign(merged['loading_fa'])
        ).mean()
        print(f"Sign concordance: {sign_concordance:.1%}")
    else:
        corr_abs = np.nan
        corr_raw = np.nan
        sign_concordance = np.nan
        print("Too few common genes for correlation")
    
    all_results.append({
        'celltype': celltype,
        'region': region,
        'fa_factor': best_factor,
        'fa_age_corr': row['best_age_corr_fa'],
        'fa_pval': row['best_pval_fa'],
        'pca_pc': best_pc,
        'pca_age_corr': row['best_age_corr_pca'],
        'pca_pval': row['best_pval_pca'],
        'n_genes_compared': len(merged),
        'top100_overlap': n_overlap,
        'top100_overlap_pct': overlap_pct,
        'loading_corr_abs': corr_abs,
        'loading_corr_raw': corr_raw,
        'sign_concordance': sign_concordance
    })
    
    # Show some example overlapping genes
    if n_overlap > 0:
        print(f"\nTop overlapping genes (by PCA loading):")
        overlap_genes = merged[merged['gene_symbol'].isin(overlap)].copy()
        overlap_genes = overlap_genes.sort_values('abs_loading_pca', ascending=False)
        
        for i, row_gene in overlap_genes.head(10).iterrows():
            gene = row_gene['gene_symbol']
            pca_load = row_gene['loading_pca']
            fa_load = row_gene['loading_fa']
            print(f"  {gene:15s} PCA={pca_load:7.3f}  FA={fa_load:7.3f}")

# Create summary
print("\n" + "="*70)
print("SUMMARY: PCA vs FA Gene Agreement")
print("="*70)

results_df = pd.DataFrame(all_results)

print(f"\nOverall statistics:")
print(f"  Mean top-100 overlap: {results_df['top100_overlap_pct'].mean():.1f}%")
print(f"  Median top-100 overlap: {results_df['top100_overlap_pct'].median():.1f}%")
print(f"  Mean loading correlation (abs): {results_df['loading_corr_abs'].mean():.3f}")
print(f"  Mean loading correlation (raw): {results_df['loading_corr_raw'].mean():.3f}")
print(f"  Mean sign concordance: {results_df['sign_concordance'].mean():.1%}")

# Sort by overlap
results_df_sorted = results_df.sort_values('top100_overlap_pct', ascending=False)

print("\n" + "-"*70)
print("Best agreements (by top-100 gene overlap):")
print("-"*70)
print(results_df_sorted[['celltype', 'region', 'top100_overlap', 
                          'loading_corr_abs', 'loading_corr_raw']].head(10).to_string(index=False))

print("\n" + "-"*70)
print("Worst agreements:")
print("-"*70)
print(results_df_sorted[['celltype', 'region', 'top100_overlap', 
                          'loading_corr_abs', 'loading_corr_raw']].tail(5).to_string(index=False))

# Check if methods agree on age correlation direction
results_df['age_corr_agree'] = (
    np.sign(results_df['fa_age_corr']) == np.sign(results_df['pca_age_corr'])
)
age_direction_agreement = results_df['age_corr_agree'].mean()
print(f"\nAge correlation direction agreement: {age_direction_agreement:.1%}")
print(f"  (Do FA and PCA both show same direction of age effect?)")

# Summary by cell type
print("\n" + "="*70)
print("Summary by Cell Type")
print("="*70)
for celltype in results_df['celltype'].unique():
    ct_df = results_df[results_df['celltype'] == celltype]
    print(f"\n{celltype}:")
    print(f"  N regions: {len(ct_df)}")
    print(f"  Mean overlap: {ct_df['top100_overlap_pct'].mean():.1f}%")
    print(f"  Mean loading corr: {ct_df['loading_corr_abs'].mean():.3f}")

# Save results
results_df_sorted.to_csv('pca_fa_comparison_summary.csv', index=False)
print("\n" + "="*70)
print("Saved: pca_fa_comparison_summary.csv")
print("="*70)

