"""
Aggregate lochNESS Age Enrichment Results

Summarizes lochNESS scores across louvain clusters per cell type × region.

Usage:
    python summarize_lochness_results.py \
        --cell-type GABAergic-neurons \
        --region ACC

Outputs:
    lochness_summary_{region}.csv
    lochness_summary_{region}.png

Author: Elli Smith
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import argparse

def summarize_lochness_results(results_dir, cell_type, region):

    search_path = os.path.join(results_dir, cell_type, f'*_{region}_lochness_scores.csv')
    score_files = glob.glob(search_path)

    print(f"Found {len(score_files)} lochNESS result files for {cell_type}, {region}")

    if len(score_files) == 0:
        print("ERROR: No lochNESS files found!")
        return

    summaries = []

    for f in sorted(score_files):
        basename = os.path.basename(f)
        louvain  = basename.split('_')[0].replace('louvain', '')
        df       = pd.read_csv(f)

        # per-animal mean lochNESS vs age regression
        animal_agg = df.groupby(['animal_id', 'age'])['lochness_score'].mean().reset_index()
        from scipy import stats
        if len(animal_agg) >= 5:
            r, p = stats.pearsonr(animal_agg['age'], animal_agg['lochness_score'])
        else:
            r, p = np.nan, np.nan

        summaries.append({
            'louvain':          louvain,
            'n_cells':          len(df),
            'n_animals':        df['animal_id'].nunique(),
            'mean_lochness':    df['lochness_score'].mean(),
            'median_lochness':  df['lochness_score'].median(),
            'std_lochness':     df['lochness_score'].std(),
            'pct_old_enriched': (df['lochness_category'] == 'old_enriched').sum() / len(df) * 100,
            'pct_young_enriched': (df['lochness_category'] == 'young_enriched').sum() / len(df) * 100,
            'pct_neutral':      (df['lochness_category'] == 'neutral').sum() / len(df) * 100,
            'n_significant':    df['lochness_significant'].sum(),
            'pct_significant':  df['lochness_significant'].sum() / len(df) * 100,
            'r_age':            r,
            'p_age':            p,
        })

    summary_df = pd.DataFrame(summaries)
    summary_df['louvain_int'] = summary_df['louvain'].astype(int)
    summary_df = summary_df.sort_values('louvain_int').drop(columns='louvain_int')

    print(f"\n{'='*70}")
    print(f"lochNESS SUMMARY: {cell_type}, {region}")
    print(f"{'='*70}")
    print(f"  Louvain clusters: {len(summary_df)}")
    print(f"  Total cells: {summary_df['n_cells'].sum():,}")
    print(f"  Mean lochNESS: {summary_df['mean_lochness'].mean():.3f} ± {summary_df['mean_lochness'].std():.3f}")
    print(f"  Old-enriched: {summary_df['pct_old_enriched'].mean():.1f}%")
    print(f"  Young-enriched: {summary_df['pct_young_enriched'].mean():.1f}%")
    print(f"  Significant: {summary_df['pct_significant'].mean():.1f}%")
    print(f"\nTop 5 by % significant:")
    print(summary_df.nlargest(5, 'pct_significant')[['louvain', 'n_cells', 'pct_significant', 'mean_lochness', 'r_age', 'p_age']].to_string(index=False))

    out_dir = os.path.join(results_dir, cell_type)
    os.makedirs(out_dir, exist_ok=True)

    csv_file = os.path.join(out_dir, f'lochness_summary_{region}.csv')
    summary_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved: {csv_file}")

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    louv_labels = summary_df['louvain'].astype(str)
    x = np.arange(len(summary_df))

    ax = axes[0, 0]
    colors = ['#E24B4A' if v > 0 else '#378ADD' for v in summary_df['mean_lochness']]
    ax.bar(x, summary_df['mean_lochness'], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(louv_labels, rotation=45)
    ax.set_xlabel('louvain'); ax.set_ylabel('mean lochNESS')
    ax.set_title('mean lochNESS by cluster')

    ax = axes[0, 1]
    ax.bar(x, summary_df['pct_significant'], edgecolor='black', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(louv_labels, rotation=45)
    ax.set_xlabel('louvain'); ax.set_ylabel('% significant cells')
    ax.set_title('significant age enrichment by cluster')

    ax = axes[1, 0]
    ax.bar(x, summary_df['pct_young_enriched'], label='young-enriched', color='#378ADD', edgecolor='black')
    ax.bar(x, summary_df['pct_neutral'], bottom=summary_df['pct_young_enriched'],
           label='neutral', color='#888780', edgecolor='black')
    ax.bar(x, summary_df['pct_old_enriched'],
           bottom=summary_df['pct_young_enriched'] + summary_df['pct_neutral'],
           label='old-enriched', color='#E24B4A', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(louv_labels, rotation=45)
    ax.set_xlabel('louvain'); ax.set_ylabel('% cells')
    ax.set_title('category distribution')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sig_mask = summary_df['p_age'] < 0.05
    ax.scatter(summary_df.loc[~sig_mask, 'mean_lochness'],
               summary_df.loc[~sig_mask, 'pct_significant'],
               s=80, edgecolor='black', alpha=0.6, color='gray', label='n.s.')
    ax.scatter(summary_df.loc[sig_mask, 'mean_lochness'],
               summary_df.loc[sig_mask, 'pct_significant'],
               s=80, edgecolor='black', alpha=0.9, color='#E24B4A', label='p<0.05')
    for _, row in summary_df.iterrows():
        ax.annotate(row['louvain'], (row['mean_lochness'], row['pct_significant']), fontsize=7)
    ax.set_xlabel('mean lochNESS'); ax.set_ylabel('% significant')
    ax.set_title('mean lochNESS vs % significant')
    ax.legend(fontsize=8)

    plt.suptitle(f'lochNESS summary: {cell_type}, {region}')
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f'lochness_summary_{region}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_file}")

    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-type',   required=True)
    parser.add_argument('--region',      required=True)
    parser.add_argument('--results-dir', default='/scratch/easmit31/factor_analysis/lochness_pca')
    args = parser.parse_args()
    summarize_lochness_results(args.results_dir, args.cell_type, args.region)
