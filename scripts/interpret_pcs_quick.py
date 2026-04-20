#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats

# Load a PCA result
pca_data = pd.read_csv('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_Glutamatergic_M1.csv')

print("="*70)
print("PC INTERPRETATION: Glutamatergic M1")
print("="*70)

# Correlate each PC with metadata
for i in range(1, 11):
    pc = f'PC{i}'
    
    print(f"\n{pc}:")
    
    # Age
    r_age, p_age = stats.pearsonr(pca_data['age'], pca_data[pc])
    if p_age < 0.05:
        print(f"  ✓ AGE: r={r_age:.3f}, p={p_age:.3e}")
    
    # Sex
    males = pca_data[pca_data['sex'] == 'M'][pc]
    females = pca_data[pca_data['sex'] == 'F'][pc]
    t, p_sex = stats.ttest_ind(males, females)
    if p_sex < 0.05:
        print(f"  ✓ SEX: M={males.mean():.2f}, F={females.mean():.2f}, p={p_sex:.3e}")
    
    # If nothing significant
    if p_age >= 0.05 and p_sex >= 0.05:
        print(f"  → Technical variation / Other biological process")

print("\n" + "="*70)

