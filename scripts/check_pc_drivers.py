#!/usr/bin/env python
"""
Check what drives each PC in cell type-region specific PCAs
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob

print("="*70)
print("ANALYZING PC DRIVERS")
print("="*70)

# Get all PCA files
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')
print(f"\nFound {len(pca_files)} PCA files")

results = []

for pca_file in pca_files:
    # Extract cell type and region from filename
    basename = pca_file.split('/')[-1]
    parts = basename.replace('pca_analysis_', '').replace('.csv', '').split('_')
    
    if len(parts) < 2:
        continue
    
    region = parts[-1]
    cell_type = '_'.join(parts[:-1])
    
    try:
        # Load PC scores
        df = pd.read_csv(pca_file)
        
        if df.empty:
            print(f"SKIPPING (empty): {cell_type} - {region}")
            continue
        
        # Get PC columns
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        
        if len(pc_cols) == 0:
            print(f"SKIPPING (no PCs): {cell_type} - {region}")
            continue
        
        print(f"{cell_type} - {region}: {len(df)} animals, {len(pc_cols)} PCs")
        
        # For each PC, test correlation with age and sex
        for pc in pc_cols:
            # Age correlation
            age_corr, age_pval = stats.spearmanr(df['age'], df[pc])
            
            # Sex effect (t-test)
            if 'sex' in df.columns and df['sex'].nunique() == 2:
                sexes = df['sex'].unique()
                group1 = df[df['sex'] == sexes[0]][pc]
                group2 = df[df['sex'] == sexes[1]][pc]
                sex_tstat, sex_pval = stats.ttest_ind(group1, group2)
            else:
                sex_tstat, sex_pval = np.nan, np.nan
            
            # Variance explained by this PC
            var_explained = df[pc].var() / sum([df[p].var() for p in pc_cols])
            
            results.append({
                'cell_type': cell_type,
                'region': region,
                'PC': pc,
                'variance_explained': var_explained,
                'age_corr': age_corr,
                'age_pval': age_pval,
                'sex_tstat': sex_tstat,
                'sex_pval': sex_pval,
                'n_animals': len(df)
            })
    
    except Exception as e:
        print(f"ERROR loading {pca_file}: {e}")
        continue

# Convert to dataframe
results_df = pd.DataFrame(results)

# Save full results
results_df.to_csv('/scratch/easmit31/factor_analysis/pc_drivers_summary.csv', index=False)
print(f"\n\nSaved: /scratch/easmit31/factor_analysis/pc_drivers_summary.csv")

# Show summary
print("\n" + "="*70)
print("SUMMARY: Which PCs capture AGE?")
print("="*70)

# Get significant age associations
sig_age = results_df[results_df['age_pval'] < 0.05].sort_values('age_corr', key=abs, ascending=False)
print(f"\nSignificant age correlations (p < 0.05): {len(sig_age)}")
print("\nTop 20 by correlation strength:")
print(sig_age[['cell_type', 'region', 'PC', 'age_corr', 'age_pval', 'variance_explained']].head(20).to_string())

print("\n" + "="*70)
print("SUMMARY: Which PCs capture SEX?")
print("="*70)

# Get significant sex effects
sig_sex = results_df[results_df['sex_pval'] < 0.05].sort_values('sex_pval')
print(f"\nSignificant sex effects (p < 0.05): {len(sig_sex)}")
print("\nTop 20:")
print(sig_sex[['cell_type', 'region', 'PC', 'sex_tstat', 'sex_pval', 'variance_explained']].head(20).to_string())

print("\n" + "="*70)
print("SUMMARY: Variance explained by each PC")
print("="*70)

# Average variance explained per PC across all analyses
avg_var = results_df.groupby('PC')['variance_explained'].agg(['mean', 'std', 'min', 'max'])
print("\nAverage variance explained:")
print(avg_var.to_string())

print("\n" + "="*70)
print("DONE!")
print("="*70)
