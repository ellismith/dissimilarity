import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def compare_raw_vs_zscore(base_path, louvain, region, min_age=1.0):
    """
    Compare raw vs z-scored dissimilarity analysis results
    """
    
    print(f"Comparing: Louvain {louvain}, Region {region}")
    print(f"{'='*70}\n")
    
    # Find files with flexible naming
    # Raw files might be: "astrocytes_louvain7_EC_minage1.0_*" or "louvain7_EC_minage1.0_*"
    # Zscore files should be: "louvain7_EC_minage1.0_zscore_*"
    
    # Find raw validation file
    raw_pattern1 = os.path.join(base_path, f"*louvain{louvain}_{region}_minage{min_age}_validation_summary.csv")
    raw_pattern2 = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_validation_summary.csv")
    
    raw_files = glob.glob(raw_pattern1)
    if not raw_files:
        raw_files = glob.glob(raw_pattern2)
    
    # Exclude zscore files
    raw_files = [f for f in raw_files if 'zscore' not in f]
    
    if not raw_files:
        print(f"ERROR: Could not find raw validation file")
        print(f"  Searched: {raw_pattern1}")
        return
    
    raw_validation = raw_files[0]
    print(f"Found raw validation: {os.path.basename(raw_validation)}")
    
    # Find zscore validation file
    zscore_validation = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_zscore_validation_summary.csv")
    
    if not os.path.exists(zscore_validation):
        print(f"ERROR: Could not find zscore validation file")
        print(f"  Expected: {zscore_validation}")
        return
    
    print(f"Found zscore validation: {os.path.basename(zscore_validation)}")
    
    # Find KNN files
    raw_knn_pattern = os.path.join(base_path, f"*louvain{louvain}_{region}_minage{min_age}_knn_analysis_k10.csv")
    raw_knn_files = glob.glob(raw_knn_pattern)
    raw_knn_files = [f for f in raw_knn_files if 'zscore' not in f]
    
    if not raw_knn_files:
        print(f"ERROR: Could not find raw KNN file")
        return
    
    raw_knn = raw_knn_files[0]
    print(f"Found raw KNN: {os.path.basename(raw_knn)}")
    
    zscore_knn = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_zscore_knn_analysis_k10.csv")
    
    if not os.path.exists(zscore_knn):
        print(f"ERROR: Could not find zscore KNN file")
        return
    
    print(f"Found zscore KNN: {os.path.basename(zscore_knn)}\n")
    
    # Load data
    raw_val = pd.read_csv(raw_validation).iloc[0]
    zscore_val = pd.read_csv(zscore_validation).iloc[0]
    
    raw_knn_df = pd.read_csv(raw_knn)
    zscore_knn_df = pd.read_csv(zscore_knn)
    
    # Create comparison table
    comparison = {
        'Metric': [],
        'Raw Expression': [],
        'Z-Scored': [],
        'Difference': [],
        'Interpretation': []
    }
    
    # Animal effects
    comparison['Metric'].append('Same/Diff Animal Ratio')
    comparison['Raw Expression'].append(f"{raw_val['ratio_diff_to_same']:.3f}")
    comparison['Z-Scored'].append(f"{zscore_val['ratio_diff_to_same']:.3f}")
    diff = zscore_val['ratio_diff_to_same'] - raw_val['ratio_diff_to_same']
    comparison['Difference'].append(f"{diff:+.3f}")
    comparison['Interpretation'].append('Lower = less animal effect' if diff < 0 else 'Higher = more animal effect')
    
    # Meaningfulness of distances
    comparison['Metric'].append('Random/NN Distance Ratio')
    comparison['Raw Expression'].append(f"{raw_val['ratio_random_to_nn']:.3f}")
    comparison['Z-Scored'].append(f"{zscore_val['ratio_random_to_nn']:.3f}")
    diff = zscore_val['ratio_random_to_nn'] - raw_val['ratio_random_to_nn']
    comparison['Difference'].append(f"{diff:+.3f}")
    comparison['Interpretation'].append('Lower = less structure' if diff < 0 else 'Higher = more structure')
    
    # Animal mixing
    comparison['Metric'].append('% Different Animal Neighbors')
    comparison['Raw Expression'].append(f"{raw_val['pct_diff_animal_neighbors']:.1f}%")
    comparison['Z-Scored'].append(f"{zscore_val['pct_diff_animal_neighbors']:.1f}%")
    diff = zscore_val['pct_diff_animal_neighbors'] - raw_val['pct_diff_animal_neighbors']
    comparison['Difference'].append(f"{diff:+.1f}%")
    comparison['Interpretation'].append('Higher = better mixing')
    
    # Age differences
    comparison['Metric'].append('Mean Age Diff to NN (years)')
    raw_age = raw_val.get('mean_age_diff_nn', np.nan)
    zscore_age = zscore_val.get('mean_age_diff_nn', np.nan)
    comparison['Raw Expression'].append(f"{raw_age:.2f}" if not pd.isna(raw_age) else 'N/A')
    comparison['Z-Scored'].append(f"{zscore_age:.2f}" if not pd.isna(zscore_age) else 'N/A')
    if not pd.isna(raw_age) and not pd.isna(zscore_age):
        diff = zscore_age - raw_age
        comparison['Difference'].append(f"{diff:+.2f}")
        comparison['Interpretation'].append('Higher = weaker age clustering')
    else:
        comparison['Difference'].append('N/A')
        comparison['Interpretation'].append('N/A')
    
    # Distance scales
    comparison['Metric'].append('Mean NN Distance')
    comparison['Raw Expression'].append(f"{raw_val['mean_nn_distance']:.2f}")
    comparison['Z-Scored'].append(f"{zscore_val['mean_nn_distance']:.2f}")
    diff = zscore_val['mean_nn_distance'] - raw_val['mean_nn_distance']
    comparison['Difference'].append(f"{diff:+.2f}")
    comparison['Interpretation'].append('Different scales (not comparable)')
    
    comp_df = pd.DataFrame(comparison)
    
    print("KEY METRICS COMPARISON:")
    print(f"{'='*70}")
    print(comp_df.to_string(index=False))
    
    # Statistical comparison
    print(f"\n{'='*70}")
    print("DISTRIBUTION COMPARISONS:")
    print(f"{'='*70}")
    
    from scipy.stats import mannwhitneyu, ks_2samp
    
    raw_age_diffs = raw_knn_df['min_age_diff']
    zscore_age_diffs = zscore_knn_df['min_age_diff']
    
    stat, pval = mannwhitneyu(raw_age_diffs, zscore_age_diffs)
    print(f"\nAge Difference to NN:")
    print(f"  Raw:     mean={raw_age_diffs.mean():.2f}, median={raw_age_diffs.median():.2f}")
    print(f"  Z-score: mean={zscore_age_diffs.mean():.2f}, median={zscore_age_diffs.median():.2f}")
    print(f"  Mann-Whitney U test: p={pval:.4f}")
    if pval < 0.05:
        print(f"  → Distributions are SIGNIFICANTLY different")
    else:
        print(f"  → Distributions are NOT significantly different")
    
    # Key insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}")
    
    # Insight 1: Animal effects
    if zscore_val['ratio_diff_to_same'] < raw_val['ratio_diff_to_same']:
        print("✓ Z-scoring REDUCED animal effects")
        print(f"  Raw expression captures magnitude differences between animals")
        print(f"  Z-scoring removes these, focusing on expression patterns")
    else:
        print("• Z-scoring did NOT reduce animal effects")
        print(f"  Animal differences are in patterns, not just magnitude")
    
    # Insight 2: Mixing
    if zscore_val['pct_diff_animal_neighbors'] > raw_val['pct_diff_animal_neighbors']:
        print(f"\n✓ Z-scoring IMPROVED animal mixing ({zscore_val['pct_diff_animal_neighbors']:.1f}% vs {raw_val['pct_diff_animal_neighbors']:.1f}%)")
        print(f"  Better for detecting biological patterns independent of scale")
    else:
        print(f"\n• Z-scoring did not improve mixing")
    
    # Insight 3: Age effects
    if not pd.isna(raw_age) and not pd.isna(zscore_age):
        age_diff = zscore_age - raw_age
        if abs(age_diff) > 0.1:
            if age_diff > 0:
                print(f"\n✓ Z-scoring WEAKENED age clustering ({zscore_age:.2f}y vs {raw_age:.2f}y)")
                print(f"  Age effects may be driven by expression magnitude, not patterns")
            else:
                print(f"\n✓ Z-scoring STRENGTHENED age clustering ({zscore_age:.2f}y vs {raw_age:.2f}y)")
                print(f"  Age effects are in expression patterns, not just magnitude")
        else:
            print(f"\n• Age clustering unchanged by z-scoring")
            print(f"  Age effects (or lack thereof) are robust to normalization")
    
    # Insight 4: Structure
    if zscore_val['ratio_random_to_nn'] < 1.5:
        print(f"\n⚠ WARNING: Z-scored distances show weak structure (ratio={zscore_val['ratio_random_to_nn']:.2f})")
        print(f"  Z-scoring may have removed too much biologically meaningful variation")
    
    # Save comparison table
    output_csv = os.path.join(base_path, f"louvain{louvain}_{region}_minage{min_age}_raw_vs_zscore_comparison.csv")
    comp_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved comparison table: {os.path.basename(output_csv)}")
    
    return comp_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare raw vs z-scored analysis results')
    parser.add_argument('--base-path', type=str, required=True,
                       help='Path to cell type directory')
    parser.add_argument('--louvain', type=str, required=True,
                       help='Louvain cluster')
    parser.add_argument('--region', type=str, required=True,
                       help='Brain region')
    parser.add_argument('--min-age', type=float, default=1.0,
                       help='Minimum age (default: 1.0)')
    
    args = parser.parse_args()
    
    compare_raw_vs_zscore(args.base_path, args.louvain, args.region, args.min_age)
