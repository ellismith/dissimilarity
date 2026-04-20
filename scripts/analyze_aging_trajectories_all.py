#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob

# Load ALL PCA analysis files (not just significant)
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')

# Exclude summary files
pca_files = [f for f in pca_files if 'summary' not in f and 'significant' not in f]

print(f"Found {len(pca_files)} PCA analysis files")
print("\n" + "="*70)
print("TESTING ALL PCs FOR NON-LINEAR AGING")
print("="*70)

nonlinear_results = []

for filename in pca_files:
    # Parse filename
    basename = filename.split('/')[-1].replace('pca_analysis_', '').replace('.csv', '')
    parts = basename.split('_')
    
    if len(parts) < 2:
        continue
    
    cell_type = parts[0]
    region = '_'.join(parts[1:])  # Handle regions with underscores
    
    try:
        pca_data = pd.read_csv(filename)
        
        # Test ALL PCs (PC1 through PC10)
        for i in range(1, 11):
            pc_name = f'PC{i}'
            
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
            
            improvement = r2_quad - r2_lin
            
            # Test if quadratic term is significant
            # Fit with quadratic term
            X = np.column_stack([pca_data['age'], pca_data['age']**2])
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, pca_data[pc_name])
            
            nonlinear_results.append({
                'cell_type': cell_type,
                'region': region,
                'pc': pc_name,
                'age_r_linear': r_lin,
                'age_p_linear': p_lin,
                'r2_linear': r2_lin,
                'r2_quadratic': r2_quad,
                'improvement': improvement,
                'is_nonlinear': improvement > 0.05,  # >5% improvement
                'is_strong_nonlinear': improvement > 0.10  # >10% improvement
            })
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Create summary dataframe
results_df = pd.DataFrame(nonlinear_results)
results_df = results_df.sort_values('improvement', ascending=False)

print(f"\n{'='*70}")
print("SUMMARY: NON-LINEAR AGING PATTERNS (ALL PCs)")
print(f"{'='*70}")

# Overall statistics
n_total = len(results_df)
n_linear_sig = (results_df['age_p_linear'] < 0.05).sum()
n_nonlinear = (results_df['improvement'] > 0.05).sum()
n_strong_nonlinear = (results_df['improvement'] > 0.10).sum()

print(f"\nTotal PCs tested: {n_total}")
print(f"Linear age effect (p<0.05): {n_linear_sig}/{n_total} ({n_linear_sig/n_total*100:.1f}%)")
print(f"Non-linear aging (>5% R² improvement): {n_nonlinear}/{n_total} ({n_nonlinear/n_total*100:.1f}%)")
print(f"Strong non-linear (>10% improvement): {n_strong_nonlinear}/{n_total} ({n_strong_nonlinear/n_total*100:.1f}%)")

# PCs that were non-significant linearly but have non-linear patterns
linear_nonsig = results_df[results_df['age_p_linear'] >= 0.05]
nonlin_in_nonsig = (linear_nonsig['improvement'] > 0.05).sum()

print(f"\n*** KEY FINDING ***")
print(f"PCs with NO linear age effect (p≥0.05): {len(linear_nonsig)}")
print(f"  Of these, showing non-linear aging: {nonlin_in_nonsig} ({nonlin_in_nonsig/len(linear_nonsig)*100:.1f}%)")
print(f"  → Non-linearity may explain some 'non-significant' results!")

# By cell type
print("\n" + "="*70)
print("NON-LINEAR AGING BY CELL TYPE")
print("="*70)

for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = results_df[results_df['cell_type'] == ct]
    if len(ct_data) > 0:
        n_nonlin = (ct_data['improvement'] > 0.05).sum()
        n_strong = (ct_data['improvement'] > 0.10).sum()
        print(f"\n{ct}:")
        print(f"  Total PCs: {len(ct_data)}")
        print(f"  Non-linear (>5%): {n_nonlin} ({n_nonlin/len(ct_data)*100:.1f}%)")
        print(f"  Strong (>10%): {n_strong} ({n_strong/len(ct_data)*100:.1f}%)")

# Top non-linear cases
print("\n" + "="*70)
print("TOP 20 MOST NON-LINEAR AGING PATTERNS")
print("="*70)
print(results_df.nlargest(20, 'improvement')[['cell_type', 'region', 'pc', 'age_r_linear', 'age_p_linear', 'r2_linear', 'r2_quadratic', 'improvement']].to_string(index=False))

# Cases with strong non-linearity but weak linear effect
interesting = results_df[(results_df['improvement'] > 0.10) & (abs(results_df['age_r_linear']) < 0.3)]
if not interesting.empty:
    print("\n" + "="*70)
    print("HIDDEN NON-LINEAR AGING (Strong quadratic, weak linear)")
    print("="*70)
    print(interesting[['cell_type', 'region', 'pc', 'age_r_linear', 'age_p_linear', 'r2_quadratic', 'improvement']].to_string(index=False))

# Save results
results_df.to_csv('nonlinear_aging_all_pcs.csv', index=False)
print(f"\nSaved: nonlinear_aging_all_pcs.csv")

# Create visualization of top cases
print("\nCreating visualizations of top non-linear cases...")

top_cases = results_df.nlargest(9, 'improvement')

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, (idx, row) in enumerate(top_cases.iterrows()):
    if i >= 9:
        break
    
    cell_type = row['cell_type']
    region = row['region']
    pc_name = row['pc']
    
    filename = f'/scratch/easmit31/factor_analysis/csv_files/pca_analysis_{cell_type}_{region}.csv'
    pca_data = pd.read_csv(filename)
    
    ax = axes[i]
    
    # Scatter
    ax.scatter(pca_data['age'], pca_data[pc_name], alpha=0.6, s=50, c='gray')
    
    # Linear fit
    z_lin = np.polyfit(pca_data['age'], pca_data[pc_name], 1)
    p_lin = np.poly1d(z_lin)
    
    # Quadratic fit
    z_quad = np.polyfit(pca_data['age'], pca_data[pc_name], 2)
    p_quad = np.poly1d(z_quad)
    
    age_smooth = np.linspace(pca_data['age'].min(), pca_data['age'].max(), 100)
    
    ax.plot(age_smooth, p_lin(age_smooth), 'b--', linewidth=2, label='Linear', alpha=0.7)
    ax.plot(age_smooth, p_quad(age_smooth), 'r-', linewidth=3, label='Quadratic', alpha=0.9)
    
    ax.set_xlabel('Age (years)', fontsize=10)
    ax.set_ylabel(pc_name, fontsize=10)
    ax.set_title(f'{cell_type} - {region} - {pc_name}\nR²: {row["r2_linear"]:.3f} → {row["r2_quadratic"]:.3f} (+{row["improvement"]:.3f})', 
                fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Top Non-Linear Aging Patterns', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('top_nonlinear_aging_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: top_nonlinear_aging_patterns.png")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

