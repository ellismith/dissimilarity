import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cell-type', required=True)
parser.add_argument('--region',    required=True)
parser.add_argument('--louvain',   required=True)
args = parser.parse_args()

pca_path   = f'/scratch/easmit31/factor_analysis/pca_embeddings/{args.cell_type}/louvain{args.louvain}_{args.region}_pca.npy'
score_path = f'/scratch/easmit31/factor_analysis/lochness_pca/{args.cell_type}/louvain{args.louvain}_{args.region}_lochness_scores.csv'

pca    = np.load(pca_path)
scores = pd.read_csv(score_path)
age_group = scores['age_group'].values

nn = NearestNeighbors(n_neighbors=11, metric='euclidean').fit(pca)
_, inds = nn.kneighbors(pca)
inds = inds[:, 1:]

focal_idx = int(scores['lochness_score'].abs().idxmax())
neighbor_idxs = inds[focal_idx]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pca[:, 0], pca[:, 1], c='lightgray', s=3, alpha=0.3, zorder=1)

for ni in neighbor_idxs:
    color = '#E24B4A' if age_group[ni] == 'old' else '#378ADD'
    ax.scatter(pca[ni, 0], pca[ni, 1], c=color, s=80, zorder=3, edgecolors='black', linewidths=0.5)
    ax.plot([pca[focal_idx, 0], pca[ni, 0]], [pca[focal_idx, 1], pca[ni, 1]],
            color='gray', lw=0.8, alpha=0.6, zorder=2)

ax.scatter(pca[focal_idx, 0], pca[focal_idx, 1], c='yellow', s=150, zorder=4,
           edgecolors='black', linewidths=1.5, marker='*')

loch = scores.loc[focal_idx, 'lochness_score']
pval = scores.loc[focal_idx, 'lochness_pvalue']
n_old = (age_group[neighbor_idxs] == 'old').sum()
ax.set_title(f'{args.cell_type} louvain{args.louvain} {args.region}\nlochNESS={loch:.3f} p={pval:.3f} | old neighbors={n_old}/10')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

out = f'/scratch/easmit31/factor_analysis/lochness_pca/{args.cell_type}/louvain{args.louvain}_{args.region}_sanity_check.png'
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f'saved: {out}')
