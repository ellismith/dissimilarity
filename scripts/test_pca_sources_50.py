#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway

# Load PCA results with 50 components
pca_df = pd.read_csv('pca_combined_all_celltypes_50pcs.csv')

print("="*70)
print("TESTING SOURCES OF VARIATION FOR 50 PCs")
print("="*70)

results = []

for i in range(1, 51):
    pc = f'PC{i}'
    
    if pc not in pca_df.columns:
        print(f"Warning: {pc} not found in data")
        continue
    
    print(f"\n{pc}:")
    
    # 1. AGE effect
    r_age, p_age = stats.pearsonr(pca_df['age'], pca_df[pc])
    
    # 2. SEX effect
    males = pca_df[pca_df['sex'] == 'M'][pc]
    females = pca_df[pca_df['sex'] == 'F'][pc]
    t_sex, p_sex = stats.ttest_ind(males, females)
    
    # 3. CELL TYPE effect (ANOVA)
    cell_groups = [pca_df[pca_df['cell_type'] == ct][pc].values 
                   for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']]
    f_ct, p_ct = f_oneway(*cell_groups)
    
    # Calculate eta-squared for cell type
    grand_mean = pca_df[pc].mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in cell_groups])
    ss_total = sum((pca_df[pc] - grand_mean)**2)
    eta2_ct = ss_between / ss_total if ss_total > 0 else 0
    
    # 4. REGION effect (ANOVA)
    region_groups = [pca_df[pca_df['region'] == reg][pc].values 
                     for reg in pca_df['region'].unique()]
    f_reg, p_reg = f_oneway(*region_groups)
    
    # Calculate eta-squared for region
    ss_between_reg = sum([len(g) * (g.mean() - grand_mean)**2 for g in region_groups])
    eta2_reg = ss_between_reg / ss_total if ss_total > 0 else 0
    
    print(f"  Age: r={r_age:.3f}, p={p_age:.2e}")
    print(f"  Sex: p={p_sex:.2e}")
    print(f"  Cell Type: η²={eta2_ct:.3f}, p={p_ct:.2e}")
    print(f"  Region: η²={eta2_reg:.3f}, p={p_reg:.2e}")
    
    results.append({
        'PC': pc,
        'age_r': r_age,
        'age_p': p_age,
        'sex_p': p_sex,
        'celltype_eta2': eta2_ct,
        'celltype_p': p_ct,
        'region_eta2': eta2_reg,
        'region_p': p_reg
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('pc_sources_of_variation_50.csv', index=False)
print(f"\n\nSaved: pc_sources_of_variation_50.csv")
print(f"Analyzed {len(results_df)} PCs")

