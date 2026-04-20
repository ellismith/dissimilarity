#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob

# Find all PCA analysis files
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')
print(f"Found {len(pca_files)} PCA analysis files")

# Test on a few examples
examples = [
    ('Glutamatergic', 'M1'),
    ('GABAergic', 'dlPFC'),
    ('Astrocytes', 'CN'),
]

for cell_type, region in examples:
    filename = f'/scratch/easmit31/factor_analysis/csv_files/pca_analysis_{cell_type}_{region}.csv'
    
    try:
        pca_data = pd.read_csv(filename)
        
        print(f"\n{'='*70}")
        print(f"Analyzing: {cell_type} - {region}")
        print(f"{'='*70}")
        
        # For each PC, analyze trajectory
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(10):
            pc_name = f'PC{i+1}'
            if pc_name not in pca_data.columns:
                continue
                
            ax = axes[i]
            
            # Scatter plot with regression line
            ax.scatter(pca_data['age'], pca_data[pc_name], alpha=0.6)
            
            # Fit polynomial to see if non-linear
            z = np.polyfit(pca_data['age'], pca_data[pc_name], 2)
            p = np.poly1d(z)
            age_smooth = np.linspace(pca_data['age'].min(), pca_data['age'].max(), 100)
            ax.plot(age_smooth, p(age_smooth), 'r--', linewidth=2)
            
            # Stats
            r, pval = stats.pearsonr(pca_data['age'], pca_data[pc_name])
            
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(pc_name)
            ax.set_title(f'{pc_name}: r={r:.3f}, p={pval:.2e}')
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'{cell_type} - {region}: Continuous Aging Trajectories', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'aging_trajectories_{cell_type}_{region}.png', dpi=300)
        print(f"Saved: aging_trajectories_{cell_type}_{region}.png")
        
        # Test for non-linearity
        print(f"\n--- Testing for Non-Linear Aging ---")
        for i in range(10):
            pc_name = f'PC{i+1}'
            if pc_name not in pca_data.columns:
                continue
            
            # Linear fit
            slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(pca_data['age'], pca_data[pc_name])
            
            # Quadratic fit
            coeffs = np.polyfit(pca_data['age'], pca_data[pc_name], 2)
            poly = np.poly1d(coeffs)
            predicted_quad = poly(pca_data['age'])
            
            # Compare R² values
            ss_res_lin = np.sum((pca_data[pc_name] - (slope_lin * pca_data['age'] + intercept_lin))**2)
            ss_tot = np.sum((pca_data[pc_name] - pca_data[pc_name].mean())**2)
            r2_lin = 1 - (ss_res_lin / ss_tot)
            
            ss_res_quad = np.sum((pca_data[pc_name] - predicted_quad)**2)
            r2_quad = 1 - (ss_res_quad / ss_tot)
            
            if r2_quad - r2_lin > 0.05 and p_lin < 0.05:  # Meaningful improvement
                print(f"  {pc_name}: Non-linear! R²_linear={r2_lin:.3f}, R²_quad={r2_quad:.3f}")
        
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error processing {cell_type}-{region}: {e}")

print("\n=== Analysis complete! ===")

