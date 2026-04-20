#!/usr/bin/env python
"""
Visualize absolute residuals from PC~age regression for top subtype-PC combinations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import stats
import glob

print("="*70)
print("VISUALIZING REGRESSION-BASED RESIDUALS")
print("="*70)

# ------------------------------------------------------------------
# Load subtype variability results
# ------------------------------------------------------------------
results_dir = "/scratch/easmit31/factor_analysis/subtype_variability"

result_files = glob.glob(f"{results_dir}/subtype_variability_*.csv")
all_results = [pd.read_csv(f) for f in result_files]
combined = pd.concat(all_results, ignore_index=True)

# Focus on Glutamatergic and GABAergic
glut_gaba = combined[
    combined['cell_type'].isin(['Glutamatergic', 'GABAergic'])
].copy()

# Top 12 by absolute correlation
top_hits = glut_gaba.sort_values(
    'abs_dev_corr', key=abs, ascending=False
).head(12)

print("\nTop 12 subtype-region-PC combinations:")
for _, row in top_hits.iterrows():
    print(f"  {row['cell_type']} - {row['region']} - {row['subtype']} - {row['PC']}: r={row['abs_dev_corr']:.3f}")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
glut_path = '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad'
gaba_path = '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'
pc_dir = '/scratch/easmit31/factor_analysis/csv_files'
output_dir = '/scratch/easmit31/factor_analysis/subtype_variability/figures'

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

plot_idx = 0

for _, row in top_hits.iterrows():
    if plot_idx >= 12:
        break

    cell_type = row['cell_type']
    region = row['region']
    subtype = row['subtype']
    pc = row['PC']

    print(f"\nProcessing: {cell_type} - {region} - {subtype} - {pc}")

    h5ad_path = glut_path if cell_type == 'Glutamatergic' else gaba_path

    try:
        adata = sc.read_h5ad(h5ad_path, backed='r')

        obs = adata.obs[['animal_id', 'region', 'age', 'ct_louvain']].copy()
        obs['animal_id'] = obs['animal_id'].astype(str)

        mask = (
            (obs['region'] == region) &
            (obs['age'] >= 1.0) &
            (obs['ct_louvain'] == subtype)
        )
        subtype_obs = obs[mask]
        animals = subtype_obs['animal_id'].unique()

        pc_file = f'{pc_dir}/pca_analysis_{cell_type}_{region}.csv'
        pc_scores = pd.read_csv(pc_file)
        pc_scores['animal_id'] = pc_scores['animal_id'].astype(str)
        pc_scores = pc_scores[pc_scores['age'] >= 1.0]

        data = pc_scores[pc_scores['animal_id'].isin(animals)].copy()

        if len(data) < 5:
            print(f"  Skipping – only {len(data)} animals")
            continue

        ax = axes[plot_idx]

        # --------------------------------------------------------------
        # Regression: PC ~ age
        # --------------------------------------------------------------
        slope, intercept, r_pc, p_pc, _ = stats.linregress(
            data['age'], data[pc]
        )

        expected_pc = intercept + slope * data['age']
        data['abs_resid'] = np.abs(data[pc] - expected_pc)

        # --------------------------------------------------------------
        # Scatter: |residual|
        # --------------------------------------------------------------
        ax.scatter(
            data['age'], data['abs_resid'],
            c=data['age'], cmap='viridis',
            s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )

        # --------------------------------------------------------------
        # Trend of residuals vs age
        # --------------------------------------------------------------
        slope_r, intercept_r, r_resid, p_resid, _ = stats.linregress(
            data['age'], data['abs_resid']
        )

        xfit = np.array([data['age'].min(), data['age'].max()])
        yfit = intercept_r + slope_r * xfit

        ax.plot(
            xfit, yfit,
            color='blue', linewidth=2
        )

        ax.set_xlabel('Age (years)', fontsize=9)
        ax.set_ylabel(f'|{pc} − expected|', fontsize=9)

        sig_marker = '***' if row['abs_dev_fdr'] < 0.05 else ''
        ax.set_title(
            f"{cell_type[:4]}-{region}-{subtype}\n"
            f"{pc}: r={r_resid:.3f}, p={p_resid:.2e} {sig_marker}",
            fontsize=9, fontweight='bold'
        )

        ax.grid(alpha=0.3)

        print(f"  Plotted {len(data)} animals")
        plot_idx += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Remove unused axes
for i in range(plot_idx, 12):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(
    f'{output_dir}/regression_abs_residuals_top12.png',
    dpi=300, bbox_inches='tight'
)
plt.close()

print(f"\nSaved: {output_dir}/regression_abs_residuals_top12.png")
print("="*70)
print("DONE!")
print("="*70)
