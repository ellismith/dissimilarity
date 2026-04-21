#!/usr/bin/env python
"""
Analyze interindividual variability in PC scores at the subtype level.
Uses existing PCA results but filters to specific subtypes.
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze subtype-level PC score variability with age')
    parser.add_argument('--cell_type', type=str, required=True,
                        help='Cell type (e.g., Microglia, Astrocytes, Glutamatergic, GABAergic)')
    parser.add_argument('--region', type=str, required=True,
                        help='Brain region (e.g., IPP, ACC, dlPFC)')
    parser.add_argument('--h5ad_path', type=str, required=True,
                        help='Path to h5ad file containing subtype annotations')
    parser.add_argument('--pc_scores_path', type=str, required=True,
                        help='Path to CSV file with PC scores (e.g., pca_analysis_Microglia_IPP.csv)')
    parser.add_argument('--subtype_col', type=str, default='ct_louvain',
                        help='Column name containing subtype labels (default: ct_louvain)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--min_animals', type=int, default=10,
                        help='Minimum number of animals required per subtype (default: 10)')
    parser.add_argument('--min_age', type=float, default=1.0,
                        help='Minimum age to include (default: 1.0)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print(f"SUBTYPE-LEVEL PC VARIABILITY ANALYSIS")
    print(f"Cell type: {args.cell_type}")
    print(f"Region: {args.region}")
    print(f"Min age: {args.min_age}")
    print("="*70)
    
    # Load h5ad to get subtype annotations
    print(f"\nLoading h5ad file: {args.h5ad_path}")
    adata = sc.read_h5ad(args.h5ad_path, backed='r')
    
    # Check if subtype column exists
    if args.subtype_col not in adata.obs.columns:
        print(f"ERROR: Column '{args.subtype_col}' not found in h5ad file!")
        print(f"Available columns: {adata.obs.columns.tolist()}")
        return
    
    # Filter to specified region
    obs_df = adata.obs[['animal_id', 'region', 'age', 'sex', args.subtype_col]].copy()
    obs_df['animal_id'] = obs_df['animal_id'].astype(str)
    obs_df['region'] = obs_df['region'].astype(str)
    
    region_mask = obs_df['region'] == args.region
    obs_df = obs_df[region_mask]
    
    if len(obs_df) == 0:
        print(f"ERROR: No cells found for region {args.region}")
        return
    
    # Filter by age
    age_mask = obs_df['age'] >= args.min_age
    print(f"Before age filter: {len(obs_df)} cells")
    obs_df = obs_df[age_mask]
    print(f"After age >= {args.min_age}: {len(obs_df)} cells")
    
    if len(obs_df) == 0:
        print(f"ERROR: No cells remaining after age filter")
        return
    
    # Get which animals have which subtypes
    print("\nIdentifying subtypes per animal...")
    animal_subtypes = obs_df.groupby('animal_id')[args.subtype_col].apply(
        lambda x: x.unique().tolist()
    ).to_dict()
    
    # Flatten to get animal-subtype pairs
    animal_subtype_list = []
    for animal_id, subtypes in animal_subtypes.items():
        for subtype in subtypes:
            animal_subtype_list.append({
                'animal_id': animal_id,
                'subtype': subtype
            })
    
    animal_subtype_df = pd.DataFrame(animal_subtype_list)
    
    # Load PC scores
    print(f"\nLoading PC scores: {args.pc_scores_path}")
    pc_scores = pd.read_csv(args.pc_scores_path)
    print(f"Loaded PC scores for {len(pc_scores)} animals")
    
    # Filter PC scores by age
    pc_scores = pc_scores[pc_scores['age'] >= args.min_age]
    print(f"After age >= {args.min_age}: {len(pc_scores)} animals")
    
    # Make sure animal_id is string in both dataframes
    pc_scores['animal_id'] = pc_scores['animal_id'].astype(str)
    animal_subtype_df['animal_id'] = animal_subtype_df['animal_id'].astype(str)
    
    # Merge subtype info with PC scores
    merged = animal_subtype_df.merge(pc_scores, on='animal_id', how='inner')
    
    if len(merged) == 0:
        print("ERROR: No matching animals found between h5ad and PC scores!")
        return
    
    print(f"Successfully matched {merged['animal_id'].nunique()} animals")
    
    # Get list of PC columns
    pc_cols = [col for col in pc_scores.columns if col.startswith('PC')]
    print(f"Found {len(pc_cols)} PCs: {pc_cols}")
    
    print(f"\nAge range: {merged['age'].min():.2f} - {merged['age'].max():.2f}")
    
    # Get subtypes with sufficient sample size
    subtype_counts = merged['subtype'].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= args.min_animals].index.tolist()
    
    print(f"\nSubtypes with >= {args.min_animals} animals:")
    for subtype in valid_subtypes:
        count = subtype_counts[subtype]
        print(f"  {subtype}: {count} animals")
    
    if len(valid_subtypes) == 0:
        print(f"ERROR: No subtypes have >= {args.min_animals} animals")
        return
    
    # Calculate variability for each subtype
    print("\n" + "="*70)
    print("CALCULATING VARIABILITY METRICS")
    print("="*70)
    
    results = []
    
    for subtype in valid_subtypes:
        subtype_data = merged[merged['subtype'] == subtype].copy()
        
        print(f"\n{subtype}:")
        print(f"  N = {len(subtype_data)} animals")
        print(f"  Age range: {subtype_data['age'].min():.2f} - {subtype_data['age'].max():.2f}")
        
        # Calculate stats for each PC
        for pc in pc_cols:
            
            # Method 1: Absolute deviation from mean correlates with age?
            # This asks: do older animals deviate more from the group mean?
            pc_mean = subtype_data[pc].mean()
            subtype_data[f'{pc}_abs_dev'] = np.abs(subtype_data[pc] - pc_mean)
            corr_dev, p_dev = stats.spearmanr(subtype_data['age'], subtype_data[f'{pc}_abs_dev'])
            
            # Method 2: Squared deviation (similar but gives more weight to outliers)
            subtype_data[f'{pc}_sq_dev'] = (subtype_data[pc] - pc_mean) ** 2
            corr_sq, p_sq = stats.spearmanr(subtype_data['age'], subtype_data[f'{pc}_sq_dev'])
            
            results.append({
                'cell_type': args.cell_type,
                'region': args.region,
                'subtype': subtype,
                'PC': pc,
                'n_animals': len(subtype_data),
                'age_range': f"{subtype_data['age'].min():.1f}-{subtype_data['age'].max():.1f}",
                'abs_dev_corr': corr_dev,
                'abs_dev_pval': p_dev,
                'sq_dev_corr': corr_sq,
                'sq_dev_pval': p_sq,
            })
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nNo results generated - insufficient data")
        return
    
    # Apply FDR correction
    print("\n" + "="*70)
    print("APPLYING FDR CORRECTION")
    print("="*70)
    
    results_df['abs_dev_fdr'] = fdrcorrection(results_df['abs_dev_pval'].fillna(1))[1]
    results_df['sq_dev_fdr'] = fdrcorrection(results_df['sq_dev_pval'].fillna(1))[1]
    
    # Summary
    sig_abs = results_df[results_df['abs_dev_fdr'] < 0.05].sort_values('abs_dev_corr', key=abs, ascending=False)
    sig_sq = results_df[results_df['sq_dev_fdr'] < 0.05].sort_values('sq_dev_corr', key=abs, ascending=False)
    
    print(f"\nSignificant results (FDR < 0.05):")
    print(f"  Absolute deviation: {len(sig_abs)} subtype-PC combinations")
    print(f"  Squared deviation: {len(sig_sq)} subtype-PC combinations")
    
    # Show top results by absolute correlation (regardless of significance)
    print("\n" + "="*70)
    print("TOP 20 RESULTS BY CORRELATION STRENGTH (regardless of FDR)")
    print("="*70)
    
    top_results = results_df.sort_values('abs_dev_corr', key=abs, ascending=False).head(20)
    
    for idx, row in top_results.iterrows():
        direction = "INCREASING" if row['abs_dev_corr'] > 0 else "DECREASING"
        sig_marker = "***" if row['abs_dev_fdr'] < 0.05 else ""
        print(f"\n{row['subtype']} - {row['PC']} {sig_marker}")
        print(f"  {direction} spread with age (N={row['n_animals']}, age {row['age_range']})")
        print(f"  r={row['abs_dev_corr']:.3f}, p={row['abs_dev_pval']:.2e}, FDR={row['abs_dev_fdr']:.2e}")
    
    if len(sig_abs) > 0:
        print("\n" + "="*70)
        print("SIGNIFICANT RESULTS (FDR < 0.05)")
        print("="*70)
        
        for idx, row in sig_abs.iterrows():
            direction = "INCREASING" if row['abs_dev_corr'] > 0 else "DECREASING"
            print(f"\n{row['subtype']} - {row['PC']}")
            print(f"  {direction} spread with age (N={row['n_animals']}, age {row['age_range']})")
            print(f"  r={row['abs_dev_corr']:.3f}, p={row['abs_dev_pval']:.2e}, FDR={row['abs_dev_fdr']:.2e}")
    
    # Save results
    import os
    output_file = os.path.join(
        args.output_dir,
        f"subtype_variability_{args.cell_type}_{args.region}.csv"
    )
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")
    
    # Print summary stats
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nCorrelation distribution (absolute deviation):")
    print(f"  Mean r: {results_df['abs_dev_corr'].mean():.3f}")
    print(f"  Median r: {results_df['abs_dev_corr'].median():.3f}")
    print(f"  Range: [{results_df['abs_dev_corr'].min():.3f}, {results_df['abs_dev_corr'].max():.3f}]")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
