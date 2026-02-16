import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_validation_results(results_dir, cell_type=None, output_dir=None):
    """
    Collect and summarize all validation results from a given directory
    
    Parameters:
    -----------
    results_dir : str
        Directory containing validation summary CSV files
    cell_type : str, optional
        Specific cell type subdirectory (e.g., 'GABAergic-neurons', 'astrocytes')
    output_dir : str, optional
        Where to save summary outputs (default: same as results_dir)
    """
    
    if output_dir is None:
        output_dir = results_dir
    
    # Find all validation summary files
    if cell_type:
        search_path = os.path.join(results_dir, cell_type, '*_validation_summary.csv')
    else:
        search_path = os.path.join(results_dir, '**', '*_validation_summary.csv')
    
    summary_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(summary_files)} validation summary files")
    
    if len(summary_files) == 0:
        print("ERROR: No validation summary files found!")
        return
    
    # Load all summaries
    all_results = []
    for f in summary_files:
        df = pd.read_csv(f)
        # Add cell type info
        cell_type_name = os.path.basename(os.path.dirname(f))
        df['cell_type'] = cell_type_name
        all_results.append(df)
    
    # Combine into one dataframe
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Parse louvain and region from base_name
    combined_df['louvain'] = combined_df['base_name'].str.extract(r'louvain(\d+)_')[0]
    combined_df['region'] = combined_df['base_name'].str.extract(r'_([A-Z]+[a-z]*(?:[A-Z][a-z]*)?)_minage')[0]
    
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total combinations analyzed: {len(combined_df)}")
    print(f"Cell types: {combined_df['cell_type'].unique()}")
    print(f"Regions: {combined_df['region'].unique()}")
    print(f"Louvain clusters: {sorted(combined_df['louvain'].unique())}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("KEY METRICS (mean ± std)")
    print(f"{'='*70}")
    
    metrics = {
        'Cells per combination': 'n_cells',
        'Animals per combination': 'n_animals',
        'Age range (min)': 'age_min',
        'Age range (max)': 'age_max',
        'Same-animal distance': 'mean_dist_same_animal',
        'Different-animal distance': 'mean_dist_diff_animal',
        'Ratio (diff/same)': 'ratio_diff_to_same',
        'NN distance': 'mean_nn_distance',
        'Random distance': 'mean_random_distance',
        'Ratio (random/NN)': 'ratio_random_to_nn',
        '% same-animal neighbors': 'pct_same_animal_neighbors',
        '% diff-animal neighbors': 'pct_diff_animal_neighbors',
    }
    
    for label, col in metrics.items():
        if col in combined_df.columns:
            mean = combined_df[col].mean()
            std = combined_df[col].std()
            print(f"  {label:30s}: {mean:8.2f} ± {std:6.2f}")
    
    # Add age metrics if available
    if 'mean_age_diff_nn' in combined_df.columns:
        print(f"\nAge-related metrics:")
        print(f"  {'Mean age diff to NN':30s}: {combined_df['mean_age_diff_nn'].mean():8.2f} ± {combined_df['mean_age_diff_nn'].std():6.2f}")
        print(f"  {'Median age diff to NN':30s}: {combined_df['median_age_diff_nn'].mean():8.2f} ± {combined_df['median_age_diff_nn'].std():6.2f}")
        print(f"  {'Mean age diff (k=10)':30s}: {combined_df['mean_age_diff_knn'].mean():8.2f} ± {combined_df['mean_age_diff_knn'].std():6.2f}")
    
    # Quality flags
    print(f"\n{'='*70}")
    print("QUALITY CHECKS")
    print(f"{'='*70}")
    
    # Flag 1: Ratio diff/same should be close to 1 (minimal animal effects)
    animal_effect_minimal = (combined_df['ratio_diff_to_same'] < 1.1).sum()
    print(f"  Minimal animal effects (ratio < 1.1): {animal_effect_minimal}/{len(combined_df)} ({animal_effect_minimal/len(combined_df)*100:.1f}%)")
    
    # Flag 2: Ratio random/NN should be > 1.5 (meaningful distances)
    meaningful_distances = (combined_df['ratio_random_to_nn'] > 1.5).sum()
    print(f"  Meaningful distances (ratio > 1.5): {meaningful_distances}/{len(combined_df)} ({meaningful_distances/len(combined_df)*100:.1f}%)")
    
    # Flag 3: High % different animal neighbors (> 70%)
    good_mixing = (combined_df['pct_diff_animal_neighbors'] > 70).sum()
    print(f"  Good animal mixing (>70% diff): {good_mixing}/{len(combined_df)} ({good_mixing/len(combined_df)*100:.1f}%)")
    
    # Summary by region
    print(f"\n{'='*70}")
    print("SUMMARY BY REGION")
    print(f"{'='*70}")
    
    region_summary = combined_df.groupby('region').agg({
        'n_cells': ['count', 'mean'],
        'ratio_diff_to_same': 'mean',
        'pct_diff_animal_neighbors': 'mean',
        'mean_nn_distance': 'mean'
    }).round(2)
    
    print(region_summary)
    
    # Save combined results
    output_file = os.path.join(output_dir, 'all_validation_results_summary.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved combined results: {output_file}")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating summary visualizations...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Animal effect (ratio diff/same)
    ax = axes[0, 0]
    ax.hist(combined_df['ratio_diff_to_same'], bins=30, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='No effect')
    ax.axvline(1.1, color='orange', linestyle='--', label='Threshold')
    ax.set_xlabel('Ratio (different-animal / same-animal distance)')
    ax.set_ylabel('Count')
    ax.set_title(f'Animal Effects\n(mean={combined_df["ratio_diff_to_same"].mean():.2f})')
    ax.legend()
    
    # Plot 2: Meaningfulness (ratio random/NN)
    ax = axes[0, 1]
    ax.hist(combined_df['ratio_random_to_nn'], bins=30, edgecolor='black')
    ax.axvline(1.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Ratio (random / nearest neighbor distance)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distance Meaningfulness\n(mean={combined_df["ratio_random_to_nn"].mean():.2f})')
    ax.legend()
    
    # Plot 3: % different animal neighbors
    ax = axes[0, 2]
    ax.hist(combined_df['pct_diff_animal_neighbors'], bins=30, edgecolor='black')
    ax.axvline(70, color='red', linestyle='--', label='Good mixing threshold')
    ax.set_xlabel('% neighbors from different animals')
    ax.set_ylabel('Count')
    ax.set_title(f'Animal Mixing\n(mean={combined_df["pct_diff_animal_neighbors"].mean():.1f}%)')
    ax.legend()
    
    # Plot 4: Age difference to NN (if available)
    if 'mean_age_diff_nn' in combined_df.columns:
        ax = axes[1, 0]
        ax.hist(combined_df['mean_age_diff_nn'], bins=30, edgecolor='black')
        ax.set_xlabel('Mean age difference to nearest neighbor (years)')
        ax.set_ylabel('Count')
        ax.set_title(f'Age Clustering\n(mean={combined_df["mean_age_diff_nn"].mean():.2f} years)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Age data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 5: Cell count distribution
    ax = axes[1, 1]
    ax.hist(combined_df['n_cells'], bins=30, edgecolor='black')
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('Count')
    ax.set_title(f'Cell Count Distribution\n(mean={combined_df["n_cells"].mean():.0f})')
    
    # Plot 6: By region comparison (ratio diff/same)
    ax = axes[1, 2]
    region_data = [combined_df[combined_df['region'] == r]['ratio_diff_to_same'].values 
                   for r in combined_df['region'].unique()]
    ax.boxplot(region_data, labels=combined_df['region'].unique())
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Region')
    ax.set_ylabel('Ratio (diff/same animal distance)')
    ax.set_title('Animal Effects by Region')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'validation_summary_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots: {plot_file}")
    plt.close()
    
    # Create a summary table for quick reference
    summary_table = combined_df[['base_name', 'cell_type', 'louvain', 'region', 'n_cells', 'n_animals',
                                  'ratio_diff_to_same', 'ratio_random_to_nn', 'pct_diff_animal_neighbors']]
    if 'mean_age_diff_nn' in combined_df.columns:
        summary_table = combined_df[['base_name', 'cell_type', 'louvain', 'region', 'n_cells', 'n_animals',
                                      'ratio_diff_to_same', 'ratio_random_to_nn', 
                                      'pct_diff_animal_neighbors', 'mean_age_diff_nn']]
    
    summary_table = summary_table.sort_values(['cell_type', 'louvain', 'region'])
    
    table_file = os.path.join(output_dir, 'validation_quick_summary.csv')
    summary_table.to_csv(table_file, index=False)
    print(f"✓ Saved quick summary table: {table_file}")
    
    print(f"\n{'='*70}")
    print("SUMMARY COMPLETE")
    print(f"{'='*70}")
    
    return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Summarize validation results across multiple analyses'
    )
    parser.add_argument('--results-dir', type=str, 
                        default='/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices',
                        help='Directory containing results')
    parser.add_argument('--cell-type', type=str, default=None,
                        help='Specific cell type to summarize (e.g., GABAergic-neurons)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results-dir)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    summarize_validation_results(
        results_dir=args.results_dir,
        cell_type=args.cell_type,
        output_dir=output_dir
    )
