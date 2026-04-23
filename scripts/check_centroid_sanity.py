"""
check_centroid_sanity.py

1. Confirms r values from pca embeddings match summary CSVs
2. Direction of r across cell types (positive vs negative)
3. FDR correction on p_mean values
4. Enrichment of significant louvains by cell type and region
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import glob

CENTROID_DIR = '/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100'
EMBED_DIR    = '/scratch/easmit31/factor_analysis/pca_embeddings'
OUT_DIR      = '/scratch/easmit31/factor_analysis'

# load all summary CSVs
rows = []
for f in glob.glob(os.path.join(CENTROID_DIR, '*_centroid_summary.csv')):
    ct = os.path.basename(f).replace('_centroid_summary.csv', '')
    if ct in ['opc', 'opc-olig']:  # skip old combined
        continue
    df = pd.read_csv(f)
    df['cell_type'] = ct
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary['louvain'] = summary['louvain'].astype(str)

# ── 1. Sanity check: recompute r for top 20 louvains ──────────────────────
print("=" * 70)
print("1. SANITY CHECK — recomputed r vs CSV r (top 20 by |r_mean|)")
print("=" * 70)

top20 = summary.nlargest(20, 'r_mean').copy()
# also include bottom 10
top20 = pd.concat([summary.nlargest(10, 'r_mean'), summary.nsmallest(10, 'r_mean')])

recomputed = []
for _, row in top20.iterrows():
    ct     = row['cell_type']
    region = row['region']
    louv   = row['louvain']
    pca_path  = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_pca.npy')
    meta_path = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_metadata.csv')
    if not os.path.exists(pca_path):
        recomputed.append({'cell_type': ct, 'region': region, 'louvain': louv,
                           'r_csv': row['r_mean'], 'r_recomputed': np.nan})
        continue

    X    = np.load(pca_path)
    meta = pd.read_csv(meta_path).reset_index(drop=True)

    animal_rows = []
    for animal in meta['animal_id'].unique():
        mask = meta['animal_id'] == animal
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        centroid = X[idxs].mean(axis=0)
        dists = np.sqrt(((X[idxs] - centroid) ** 2).sum(axis=1))
        animal_rows.append({'age': meta.loc[mask, 'age'].iloc[0], 'mean_dist': dists.mean()})

    agg = pd.DataFrame(animal_rows)
    r, _ = stats.pearsonr(agg['age'], agg['mean_dist'])
    recomputed.append({'cell_type': ct, 'region': region, 'louvain': louv,
                       'r_csv': row['r_mean'], 'r_recomputed': round(r, 4)})

rc_df = pd.DataFrame(recomputed)
rc_df['diff'] = (rc_df['r_csv'] - rc_df['r_recomputed']).abs()
print(rc_df[['cell_type','region','louvain','r_csv','r_recomputed','diff']].to_string(index=False))
print(f"\nMax difference: {rc_df['diff'].max():.6f}")

# ── 2. Direction of r by cell type ────────────────────────────────────────
print("\n" + "=" * 70)
print("2. DIRECTION OF r_mean BY CELL TYPE")
print("=" * 70)

dir_df = summary.groupby('cell_type').apply(lambda x: pd.Series({
    'n_louvains':   len(x),
    'n_positive':   (x['r_mean'] > 0).sum(),
    'n_negative':   (x['r_mean'] < 0).sum(),
    'pct_positive': (x['r_mean'] > 0).sum() / len(x) * 100,
    'mean_r':       x['r_mean'].mean(),
    'median_r':     x['r_mean'].median(),
})).reset_index()

print(dir_df.to_string(index=False))

# ── 3. FDR correction ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. FDR CORRECTION (Benjamini-Hochberg)")
print("=" * 70)

summary = summary.dropna(subset=['p_mean'])
reject, pvals_fdr, _, _ = multipletests(summary['p_mean'], method='fdr_bh')
summary['p_fdr'] = pvals_fdr
summary['sig_nominal'] = summary['p_mean'] < 0.05
summary['sig_fdr']     = reject

print(f"Total louvains: {len(summary)}")
print(f"Significant at p<0.05:     {summary['sig_nominal'].sum()} ({summary['sig_nominal'].mean()*100:.1f}%)")
print(f"Significant after FDR q<0.05: {summary['sig_fdr'].sum()} ({summary['sig_fdr'].mean()*100:.1f}%)")

# ── 4. Enrichment by cell type and region ─────────────────────────────────
print("\n" + "=" * 70)
print("4. SIGNIFICANT LOUVAINS BY CELL TYPE (FDR)")
print("=" * 70)

ct_enrich = summary.groupby('cell_type').apply(lambda x: pd.Series({
    'n_total':   len(x),
    'n_sig_fdr': x['sig_fdr'].sum(),
    'pct_sig':   x['sig_fdr'].sum() / len(x) * 100,
    'mean_r':    x['r_mean'].mean(),
})).reset_index().sort_values('pct_sig', ascending=False)

print(ct_enrich.to_string(index=False))

print("\n" + "=" * 70)
print("4b. SIGNIFICANT LOUVAINS BY REGION (FDR)")
print("=" * 70)

reg_enrich = summary.groupby('region').apply(lambda x: pd.Series({
    'n_total':   len(x),
    'n_sig_fdr': x['sig_fdr'].sum(),
    'pct_sig':   x['sig_fdr'].sum() / len(x) * 100,
    'mean_r':    x['r_mean'].mean(),
})).reset_index().sort_values('pct_sig', ascending=False)

print(reg_enrich.to_string(index=False))

# save enriched summary
out_csv = os.path.join(OUT_DIR, 'centroid_summary_with_fdr.csv')
summary.to_csv(out_csv, index=False)
print(f"\n✓ Saved: {out_csv}")

# ── Plot: direction heatmap ───────────────────────────────────────────────
CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']

import seaborn as sns

r_mat   = summary.pivot_table(index='cell_type', columns='region', values='r_mean').reindex(index=CELL_TYPES, columns=REGIONS)
sig_mat = summary.pivot_table(index='cell_type', columns='region', values='sig_fdr', aggfunc='mean').reindex(index=CELL_TYPES, columns=REGIONS) * 100

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

sns.heatmap(r_mat, ax=axes[0], cmap='RdBu_r', center=0, vmin=-0.6, vmax=0.6,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white', mask=r_mat.isna(),
            cbar_kws={'shrink': 0.8, 'label': 'mean r'})
axes[0].set_title('mean r_mean (centroid distance ~ age)\ncollapsed across louvains')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

sns.heatmap(sig_mat, ax=axes[1], cmap='Reds', vmin=0, vmax=100,
            annot=True, fmt='.0f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white', mask=sig_mat.isna(),
            cbar_kws={'shrink': 0.8, 'label': '% louvains sig (FDR)'})
axes[1].set_title('% louvains significant\n(FDR q<0.05)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.suptitle('Within-animal centroid distance ~ age', y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'centroid_direction_fdr_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")
