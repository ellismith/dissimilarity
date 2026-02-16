import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def compare_subtype_enrichment(results_dir, cell_type='GABAergic-neurons', region='HIP'):
    """
    Compare young vs old enrichment patterns across subtypes
    """
    
    # Find all lochness score files
    search_path = os.path.join(results_dir, cell_type, f'*_{region}_*_lochness_scores.csv')
    score_files = glob.glob(search_path)
    
    print(f"Analyzing {len(score_files)} GABAergic subtypes in {region}")
    print(f"{'='*70}\n")
    
    # Collect enrichment stats by subtype
    subtype_stats = []
    all_data = []
    
    for f in score_files:
        basename = os.path.basename(f)
        louvain = basename.split('_')[0].replace('louvain', '')
        
        df = pd.read_csv(f)
        df['louvain'] = louvain
        all_data.append(df)
        
        # Calculate enrichment metrics
        n_cells = len(df)
        n_young_enriched = (df['lochness_category'] == 'young_enriched').sum()
        n_old_enriched = (df['lochness_category'] == 'old_enriched').sum()
        
        # Mean lochNESS for young vs old cells
        young_cells = df[df['age_group'] == 'young']
        old_cells = df[df['age_group'] == 'old']
        
        stats = {
            'louvain': louvain,
            'n_cells': n_cells,
            'pct_young_enriched': n_young_enriched / n_cells * 100,
            'pct_old_enriched': n_old_enriched / n_cells * 100,
            'enrichment_ratio': n_young_enriched / n_old_enriched if n_old_enriched > 0 else np.nan,
            'mean_lochness': df['lochness_score'].mean(),
            'mean_lochness_young_cells': young_cells['lochness_score'].mean() if len(young_cells) > 0 else np.nan,
            'mean_lochness_old_cells': old_cells['lochness_score'].mean() if len(old_cells) > 0 else np.nan,
            'pct_significant': df['lochness_significant'].sum() / n_cells * 100,
            'n_young_cells': len(young_cells),
            'n_old_cells': len(old_cells),
            'pct_young_cells': len(young_cells) / n_cells * 100
        }
        
        subtype_stats.append(stats)
    
    stats_df = pd.DataFrame(subtype_stats)
    stats_df = stats_df.sort_values('enrichment_ratio', ascending=False)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Print summary
    print("SUBTYPES RANKED BY YOUNG/OLD ENRICHMENT RATIO:")
    print(f"{'='*70}")
    print(stats_df[['louvain', 'n_cells', 'pct_young_enriched', 'pct_old_enriched', 
                    'enrichment_ratio', 'pct_significant']].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("TOP 5 SUBTYPES WITH STRONGEST YOUNG BIAS:")
    print(f"{'='*70}")
    top5_young = stats_df.nlargest(5, 'enrichment_ratio')
    for _, row in top5_young.iterrows():
        print(f"\nLouvain {row['louvain']}:")
        print(f"  Cells: {row['n_cells']}")
        print(f"  Young-enriched: {row['pct_young_enriched']:.1f}%")
        print(f"  Old-enriched: {row['pct_old_enriched']:.1f}%")
        print(f"  Ratio: {row['enrichment_ratio']:.2f}x more young-enriched")
        print(f"  Mean lochNESS (young cells): {row['mean_lochness_young_cells']:.3f}")
        print(f"  Mean lochNESS (old cells): {row['mean_lochness_old_cells']:.3f}")
    
    print(f"\n{'='*70}")
    print("BOTTOM 5 SUBTYPES (Most Balanced or Old-Biased):")
    print(f"{'='*70}")
    bottom5 = stats_df.nsmallest(5, 'enrichment_ratio')
    for _, row in bottom5.iterrows():
        print(f"\nLouvain {row['louvain']}:")
        print(f"  Cells: {row['n_cells']}")
        print(f"  Young-enriched: {row['pct_young_enriched']:.1f}%")
        print(f"  Old-enriched: {row['pct_old_enriched']:.1f}%")
        print(f"  Ratio: {row['enrichment_ratio']:.2f}x")
    
    # Save results
    output_file = os.path.join(results_dir, cell_type, f'subtype_enrichment_comparison_{region}.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison: {output_file}")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating comparison visualizations...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Young vs Old enrichment scatter
    ax = axes[0, 0]
    scatter = ax.scatter(stats_df['pct_old_enriched'], stats_df['pct_young_enriched'],
                        s=stats_df['n_cells']/10, alpha=0.6, c=stats_df['enrichment_ratio'],
                        cmap='RdBu_r', vmin=0, vmax=4, edgecolor='black')
    
    # Add diagonal (equal enrichment)
    max_val = max(stats_df['pct_old_enriched'].max(), stats_df['pct_young_enriched'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal')
    
    # Label outliers
    for _, row in stats_df.iterrows():
        if row['enrichment_ratio'] > 2.5 or row['enrichment_ratio'] < 0.8:
            ax.annotate(row['louvain'], (row['pct_old_enriched'], row['pct_young_enriched']),
                       fontsize=8, ha='center')
    
    ax.set_xlabel('% Old-Enriched Cells')
    ax.set_ylabel('% Young-Enriched Cells')
    ax.set_title('Young vs Old Enrichment by Subtype')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Young/Old Ratio')
    
    # Plot 2: Enrichment ratio by subtype
    ax = axes[0, 1]
    colors = ['red' if r > 1.5 else 'gray' for r in stats_df['enrichment_ratio']]
    ax.barh(stats_df['louvain'].astype(str), stats_df['enrichment_ratio'], 
            color=colors, edgecolor='black')
    ax.axvline(1, color='black', linestyle='--', alpha=0.5, label='Equal')
    ax.set_xlabel('Young/Old Enrichment Ratio')
    ax.set_ylabel('Louvain Subtype')
    ax.set_title('Young Bias by Subtype')
    ax.legend()
    
    # Plot 3: Mean lochNESS for young vs old cells
    ax = axes[0, 2]
    x = np.arange(len(stats_df))
    width = 0.35
    
    ax.bar(x - width/2, stats_df['mean_lochness_young_cells'], width, 
           label='Young cells', color='blue', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, stats_df['mean_lochness_old_cells'], width,
           label='Old cells', color='red', alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Louvain Subtype')
    ax.set_ylabel('Mean lochNESS Score')
    ax.set_title('Mean lochNESS: Young vs Old Cells')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['louvain'].astype(str), rotation=45)
    ax.legend()
    
    # Plot 4: Distribution of lochNESS by subtype (violin plot for top 5)
    ax = axes[1, 0]
    top5_louvains = top5_young['louvain'].values
    top5_data = combined_df[combined_df['louvain'].isin(top5_louvains)]
    
    sns.violinplot(data=top5_data, x='louvain', y='lochness_score', 
                   hue='age_group', split=True, ax=ax, palette={'young': 'blue', 'old': 'red'})
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Louvain Subtype (Top 5 Young-Biased)')
    ax.set_ylabel('lochNESS Score')
    ax.set_title('lochNESS Distribution: Young vs Old Cells')
    
    # Plot 5: Cell count vs enrichment ratio
    ax = axes[1, 1]
    ax.scatter(stats_df['n_cells'], stats_df['enrichment_ratio'], 
              s=100, alpha=0.6, edgecolor='black')
    for _, row in stats_df.iterrows():
        ax.annotate(row['louvain'], (row['n_cells'], row['enrichment_ratio']),
                   fontsize=8, ha='center')
    ax.axhline(1, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Young/Old Enrichment Ratio')
    ax.set_title('Cell Count vs Young Bias')
    
    # Plot 6: % Significant vs enrichment ratio
    ax = axes[1, 2]
    ax.scatter(stats_df['enrichment_ratio'], stats_df['pct_significant'],
              s=stats_df['n_cells']/10, alpha=0.6, edgecolor='black')
    for _, row in stats_df.iterrows():
        if row['pct_significant'] > 9:
            ax.annotate(row['louvain'], (row['enrichment_ratio'], row['pct_significant']),
                       fontsize=8, ha='center')
    ax.axvline(1, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Young/Old Enrichment Ratio')
    ax.set_ylabel('% Significant Cells')
    ax.set_title('Young Bias vs Statistical Significance')
    
    plt.suptitle(f'Subtype Enrichment Comparison: {cell_type}, {region}', y=0.995)
    plt.tight_layout()
    
    plot_file = os.path.join(results_dir, cell_type, f'subtype_enrichment_comparison_{region}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots: {plot_file}")
    plt.close()
    
    return stats_df, combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare enrichment patterns across subtypes')
    parser.add_argument('--results-dir', type=str,
                       default='/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices',
                       help='Results directory')
    parser.add_argument('--cell-type', type=str, default='GABAergic-neurons',
                       help='Cell type')
    parser.add_argument('--region', type=str, default='HIP',
                       help='Brain region')
    
    args = parser.parse_args()
    
    compare_subtype_enrichment(args.results_dir, args.cell_type, args.region)
