import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv('/scratch/easmit31/factor_analysis/centroid_dist_age_HIP_all_celltypes.csv')

# pivot to matrix of r values
r_mat = df.pivot(index='cell_type', columns='louvain', values='r')
p_mat = df.pivot(index='cell_type', columns='louvain', values='p')

fig, ax = plt.subplots(figsize=(20, 6))

im = ax.imshow(r_mat.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
plt.colorbar(im, ax=ax, label='r (age ~ mean dist to centroid)')

# mark significant ones
for i in range(p_mat.shape[0]):
    for j in range(p_mat.shape[1]):
        p = p_mat.values[i, j]
        if pd.isna(p):
            continue
        if p < 0.01:
            ax.text(j, i, '**', ha='center', va='center', fontsize=7, color='black')
        elif p < 0.05:
            ax.text(j, i, '*', ha='center', va='center', fontsize=8, color='black')

ax.set_xticks(range(len(r_mat.columns)))
ax.set_xticklabels(r_mat.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(r_mat.index)))
ax.set_yticklabels(r_mat.index, fontsize=9)
ax.set_xlabel('Louvain cluster')
ax.set_title('HIP - centroid distance vs age (r values)\n* p<0.05, ** p<0.01')

plt.tight_layout()
plt.savefig('/scratch/easmit31/factor_analysis/centroid_dist_summary_HIP.png', dpi=150)
plt.close()
print("Done.")
