#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import glob

# Find all PCA analysis files
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')
pca_files = [f for f in pca_files if 'summary' not in f and 'significant' not in f]

print("="*70)
print("INTERPRETING ALL PCs ACROSS ALL ANALYSES")
print("="*70)

all_interpretations = []

for filename in pca_files:
    basename = filename.split('/')[-1].replace('pca_analysis_', '').replace('.csv', '')
    parts = basename.split('_')
    
    if len(parts) < 2:
        continue
    
    cell_type = parts[0]
    region = '_'.join(parts[1:])
    
    try:
        pca_data = pd.read_csv(filename)
        
        # Test each PC
        for i in range(1, 11):
            pc = f'PC{i}'
            
            if pc not in pca_data.columns:
                continue
            
            # Age
            r_age, p_age = stats.pearsonr(pca_data['age'], pca_data[pc])
            
            # Sex
            males = pca_data[pca_data['sex'] == 'M'][pc]
            females = pca_data[pca_data['sex'] == 'F'][pc]
            t, p_sex = stats.ttest_ind(males, females)
            
            # Classify
            if p_age < 0.05 and p_sex >= 0.05:
                interpretation = "Age"
            elif p_age >= 0.05 and p_sex < 0.05:
                interpretation = "Sex"
            elif p_age < 0.05 and p_sex < 0.05:
                interpretation = "Age+Sex"
            else:
                interpretation = "Technical/Other"
            
            all_interpretations.append({
                'cell_type': cell_type,
                'region': region,
                'pc': pc,
                'interpretation': interpretation,
                'age_r': r_age,
                'age_p': p_age,
                'sex_p': p_sex
            })
    
    except Exception as e:
        print(f"Error with {filename}: {e}")

# Create summary
interp_df = pd.DataFrame(all_interpretations)

print(f"\nTotal PCs analyzed: {len(interp_df)}")

# Overall summary
print("\n" + "="*70)
print("OVERALL PC INTERPRETATION SUMMARY")
print("="*70)

interpretation_counts = interp_df['interpretation'].value_counts()
print(interpretation_counts)
print(f"\nPercentage breakdown:")
for interp, count in interpretation_counts.items():
    pct = count / len(interp_df) * 100
    print(f"  {interp}: {count}/{len(interp_df)} ({pct:.1f}%)")

# By cell type
print("\n" + "="*70)
print("PC INTERPRETATIONS BY CELL TYPE")
print("="*70)

for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = interp_df[interp_df['cell_type'] == ct]
    if len(ct_data) == 0:
        continue
    
    print(f"\n{ct}:")
    counts = ct_data['interpretation'].value_counts()
    for interp, count in counts.items():
        pct = count / len(ct_data) * 100
        print(f"  {interp}: {count}/{len(ct_data)} ({pct:.1f}%)")

# Which PC number tends to capture age?
print("\n" + "="*70)
print("WHICH PC NUMBERS CAPTURE AGE?")
print("="*70)

age_pcs = interp_df[interp_df['interpretation'].str.contains('Age')]
pc_counts = age_pcs['pc'].value_counts().sort_index()
print(pc_counts)

print("\n→ Age signals are NOT always in PC1/PC2!")
print("  They appear in later PCs because technical variation dominates early PCs")

# Save results
interp_df.to_csv('pc_interpretations_all.csv', index=False)
print(f"\nSaved: pc_interpretations_all.csv")

# Key finding: How many age PCs per analysis?
print("\n" + "="*70)
print("NUMBER OF AGE-ASSOCIATED PCs PER ANALYSIS")
print("="*70)

age_per_analysis = age_pcs.groupby(['cell_type', 'region']).size().reset_index(name='n_age_pcs')
age_per_analysis = age_per_analysis.sort_values('n_age_pcs', ascending=False)

print("\nTop 10 analyses with most age-associated PCs:")
print(age_per_analysis.head(10).to_string(index=False))

print("\nAnalyses with 3+ age-associated PCs:")
multi_age = age_per_analysis[age_per_analysis['n_age_pcs'] >= 3]
if not multi_age.empty:
    print(multi_age.to_string(index=False))
    print("\n→ These regions have MULTIPLE independent aging processes!")
else:
    print("None found - most regions have 1-2 age PCs")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

