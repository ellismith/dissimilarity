"""
Aggregate lochNESS Age Enrichment Results (continuous age version)

Summarizes neighborhood age enrichment scores across louvain clusters
per cell type × region.

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
from scipy import stats

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

        # per-animal mean neighbor age vs animal age regression
        animal_agg = df.groupby(['animal_id', 'age'])['neighbor_mean_age'].mean().reset_index()
        if len(animal_agg) >= 5:
            r_age, p_age = stats.pearsonr(animal_agg['age'], animal_agg['neighbor_mean_age'])
        else:
            r_age, p_age = np.nan, np.nan

        # per-animal mean zscore vs age
        animal_agg_z = df.groupby(['animal_id', 'age'])['neighbor_age_zscore'].mean().reset_index()
        if len(animal_agg_z) >= 5:
            r_zscore, p_zscore = stats.pearsonr(animal_agg_z['age'], animal_agg_z['neighbor_age_zscore'])
        else:
            r_zscore, p_zscore = np.nan, np.nan

        summaries.append({
            'louvain':              louvain,
            'n_cells':              len(df),
            'n_animals':            df['animal_id'].nunique(),
            'mean_neighbor_age':    df['neighbor_mean_age'].mean(),
            'mean_zscore':          df['neighbor_age_zscore'].mean(),
            'std_zscore':           df['neighbor_age_zscore'].std(),
            'r_age':                round(r_age, 4) if not np.isnan(r_age) else np.nan,
            'p_age':                round(p_age, 4) if not np.isnan(p_age) else np.nan,
            'r_zscore':             round(r_zscore, 4) if not np.isnan(r_zscore) else np.nan,
            'p_zscore':             round(p_zscore, 4) if not np.isnan(p_zscore) else np.nan,
        })

    summary_df = pd.DataFrame(summaries)
    summary_df['louvain_int'] = summary_df['louvain'].astype(int)
    summary_df = summary_df.sort_values('louvain_int').drop(columns='louvain_int')

    print(f"\n{'='*70}")
    print(f"lochNESS SUMMARY: {cell_type}, {region}")
    print(f"{'='*70}")
    print(f"  Louvain clusters: {len(summary_df)}")
    print(f"  Total cells: {summary_df['n_cells'].sum():,}")
    print(f"  Mean neighbor age zscore: {summary_df['mean_zscore'].mean():.3f} ± {summary_df['mean_zscore'].std():.3f}")
    n_sig = (summary_df['p_age'] < 0.05).sum()
    print(f"  Louvains with sig age r (p<0.05): {n_sig}/{len(summary_df)}")
    print(f"\nTop 5 by |r_age|:")
    print(summary_df.dropna(subset=['r_age']).reindex(
        summary_df.dropna(subset=['r_age'])['r_age'].abs().sort_values(ascending=False).index
    )[['louvain','n_cells','r_age','p_age','r_zscore','p_zscore']].head(5).to_string(index=False))

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
    colors = ['#E24B4A' if v > 0 else '#378ADD' for v in summary_df['r_age']]
    ax.bar(x, summary_df['r_age'], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(louv_labels, rotation=45)
    ax.set_xlabel('louvain'); ax.set_ylabel('Pearson r (neighbor age vs cell age)')
    ax.set_title('age clustering r by cluster')

    ax = axes[0, 1]
    colors2 = ['#E24B4A' if v > 0 else '#378ADD' for v in summary_df['mean_zscore']]
    ax.bar(x, summary_df['mean_zscore'], color=colors2, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(louv_labels, rotation=45)
    ax.set_xlabel('louvain'); ax.set_ylabel('mean neighbor age z-score')
    ax.set_title('mean z-score by cluster')

    ax = axes[1, 0]
    sig_mask = summary_df['p_age'] < 0.05
    ax.scatter(summary_df.loc[~sig_mask, 'r_age'],
               summary_df.loc[~sig_mask, 'mean_zscore'],
               s=80, edgecolor='black', alpha=0.6, color='gray', label='n.s.')
    ax.scatter(summary_df.loc[sig_mask, 'r_age'],
               summary_df.loc[sig_mask, 'mean_zscore'],
               s=80, edgecolor='black', alpha=0.9, color='#E24B4A', label='p<0.05')
    for _, row in summary_df.iterrows():
        ax.annotate(row['louvain'], (row['r_age'], row['mean_zscore']), fontsize=7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('r_age'); ax.set_ylabel('mean z-score')
    ax.set_title('r_age vs mean z-score')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(summary_df['n_cells'], summary_df['r_age'].abs(),
               s=80, edgecolor='black', alpha=0.6)
    for _, row in summary_df.iterrows():
        ax.annotate(row['louvain'], (row['n_cells'], abs(row['r_age'])), fontsize=7)
    ax.set_xlabel('n cells'); ax.set_ylabel('|r_age|')
    ax.set_title('cell count vs effect size')

    plt.suptitle(f'lochNESS summary (continuous age): {cell_type}, {region}')
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
