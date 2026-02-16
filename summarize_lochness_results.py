import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

def summarize_lochness_results(results_dir, cell_type='GABAergic-neurons', region='HIP'):
    """Summarize lochNESS results across all louvain clusters"""
    
    # Find all lochness score files
    search_path = os.path.join(results_dir, cell_type, f'*_{region}_*_lochness_scores.csv')
    score_files = glob.glob(search_path)
    
    print(f"Found {len(score_files)} lochNESS result files for {cell_type}, {region}")
    
    if len(score_files) == 0:
        print("ERROR: No lochNESS files found!")
        return
    
    # Collect summary stats for each louvain
    summaries = []
    
    for f in score_files:
        # Extract louvain from filename
        basename = os.path.basename(f)
        louvain = basename.split('_')[0].replace('louvain', '')
        
        # Load results
        df = pd.read_csv(f)
        
        # Calculate summary stats
        summary = {
            'louvain': louvain,
            'n_cells': len(df),
            'mean_lochness': df['lochness_score'].mean(),
            'median_lochness': df['lochness_score'].median(),
            'std_lochness': df['lochness_score'].std(),
            'min_lochness': df['lochness_score'].min(),
            'max_lochness': df['lochness_score'].max(),
            'n_old_enriched': (df['lochness_category'] == 'old_enriched').sum(),
            'n_young_enriched': (df['lochness_category'] == 'young_enriched').sum(),
            'n_neutral': (df['lochness_category'] == 'neutral').sum(),
            'pct_old_enriched': (df['lochness_category'] == 'old_enriched').sum() / len(df) * 100,
            'pct_young_enriched': (df['lochness_category'] == 'young_enriched').sum() / len(df) * 100,
            'pct_neutral': (df['lochness_category'] == 'neutral').sum() / len(df) * 100,
            'n_significant': df['lochness_significant'].sum(),
            'pct_significant': df['lochness_significant'].sum() / len(df) * 100
        }
        
        summaries.append(summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values('louvain')
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"lochNESS SUMMARY: {cell_type}, {region}")
    print(f"{'='*70}")
    
    print(f"\nOverall statistics (across {len(summary_df)} louvain clusters):")
    print(f"  Total cells analyzed: {summary_df['n_cells'].sum():,}")
    print(f"  Mean lochNESS: {summary_df['mean_lochness'].mean():.3f} ± {summary_df['mean_lochness'].std():.3f}")
    print(f"  Median lochNESS: {summary_df['median_lochness'].mean():.3f} ± {summary_df['median_lochness'].std():.3f}")
    
    print(f"\nEnrichment categories (mean across clusters):")
    print(f"  Old-enriched: {summary_df['pct_old_enriched'].mean():.1f}% ± {summary_df['pct_old_enriched'].std():.1f}%")
    print(f"  Young-enriched: {summary_df['pct_young_enriched'].mean():.1f}% ± {summary_df['pct_young_enriched'].std():.1f}%")
    print(f"  Neutral: {summary_df['pct_neutral'].mean():.1f}% ± {summary_df['pct_neutral'].std():.1f}%")
    
    print(f"\nStatistical significance:")
    print(f"  Significant cells: {summary_df['pct_significant'].mean():.1f}% ± {summary_df['pct_significant'].std():.1f}%")
    
    print(f"\nTop 5 clusters with most age enrichment (by % significant):")
    top5 = summary_df.nlargest(5, 'pct_significant')[['louvain', 'n_cells', 'pct_significant', 'mean_lochness']]
    print(top5.to_string(index=False))
    
    # Save summary
    output_file = os.path.join(results_dir, cell_type, f'lochness_summary_{region}.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved summary: {output_file}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean lochNESS by louvain
    ax = axes[0, 0]
    ax.bar(summary_df['louvain'].astype(str), summary_df['mean_lochness'], edgecolor='black')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Louvain Cluster')
    ax.set_ylabel('Mean lochNESS Score')
    ax.set_title('Mean lochNESS by Cluster')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: % Significant cells by louvain
    ax = axes[0, 1]
    ax.bar(summary_df['louvain'].astype(str), summary_df['pct_significant'], edgecolor='black')
    ax.set_xlabel('Louvain Cluster')
    ax.set_ylabel('% Significant Cells (p<0.05)')
    ax.set_title('Significant Age Enrichment by Cluster')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Enrichment categories stacked bar
    ax = axes[1, 0]
    x = np.arange(len(summary_df))
    width = 0.8
    
    ax.bar(x, summary_df['pct_young_enriched'], width, label='Young-enriched', color='blue', edgecolor='black')
    ax.bar(x, summary_df['pct_neutral'], width, bottom=summary_df['pct_young_enriched'], 
           label='Neutral', color='gray', edgecolor='black')
    ax.bar(x, summary_df['pct_old_enriched'], width, 
           bottom=summary_df['pct_young_enriched'] + summary_df['pct_neutral'],
           label='Old-enriched', color='red', edgecolor='black')
    
    ax.set_xlabel('Louvain Cluster')
    ax.set_ylabel('% of Cells')
    ax.set_title('Cell Category Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['louvain'].astype(str), rotation=45)
    ax.legend()
    
    # Plot 4: Cell count vs % significant
    ax = axes[1, 1]
    ax.scatter(summary_df['n_cells'], summary_df['pct_significant'], s=100, alpha=0.6, edgecolor='black')
    for idx, row in summary_df.iterrows():
        ax.annotate(row['louvain'], (row['n_cells'], row['pct_significant']), 
                   fontsize=8, ha='center')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('% Significant Cells')
    ax.set_title('Cell Count vs Age Enrichment')
    
    plt.suptitle(f'lochNESS Summary: {cell_type}, {region}', y=0.995)
    plt.tight_layout()
    
    plot_file = os.path.join(results_dir, cell_type, f'lochness_summary_{region}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plots: {plot_file}")
    plt.close()
    
    return summary_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize lochNESS results')
    parser.add_argument('--results-dir', type=str,
                       default='/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices',
                       help='Results directory')
    parser.add_argument('--cell-type', type=str, default='GABAergic-neurons',
                       help='Cell type')
    parser.add_argument('--region', type=str, default='HIP',
                       help='Brain region')
    
    args = parser.parse_args()
    
    summarize_lochness_results(args.results_dir, args.cell_type, args.region)
