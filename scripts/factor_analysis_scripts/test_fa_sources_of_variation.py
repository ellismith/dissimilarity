#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway

# Load FA results
fa_df = pd.read_csv('fa_combined_all_celltypes_50factors.csv')

print("="*70)
print("TESTING WHAT EACH FACTOR CAPTURES")
print("="*70)

results = []

for i in range(1, 51):
    factor = f'Factor{i}'
    
    if factor not in fa_df.columns:
        continue
    
    print(f"\n{factor}:")
    
    # 1. AGE effect
    r_age, p_age = stats.pearsonr(fa_df['age'], fa_df[factor])
    
    # 2. SEX effect
    males = fa_df[fa_df['sex'] == 'M'][factor]
    females = fa_df[fa_df['sex'] == 'F'][factor]
    t_sex, p_sex = stats.ttest_ind(males, females)
    
    # 3. CELL TYPE effect (ANOVA)
    cell_groups = [fa_df[fa_df['cell_type'] == ct][factor].values 
                   for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']]
    f_ct, p_ct = f_oneway(*cell_groups)
    
    # Calculate eta-squared (effect size) for cell type
    grand_mean = fa_df[factor].mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in cell_groups])
    ss_total = sum((fa_df[factor] - grand_mean)**2)
    eta2_ct = ss_between / ss_total if ss_total > 0 else 0
    
    # 4. REGION effect (ANOVA)
    regions = fa_df['region'].unique()
    region_groups = [fa_df[fa_df['region'] == r][factor].values 
                     for r in regions if len(fa_df[fa_df['region'] == r]) > 0]
    f_reg, p_reg = f_oneway(*region_groups)
    
    # Calculate eta-squared for region
    ss_between_reg = sum([len(g) * (g.mean() - grand_mean)**2 for g in region_groups])
    eta2_reg = ss_between_reg / ss_total if ss_total > 0 else 0
    
    # Print results
    sig_markers = []
    if p_age < 0.05:
        print(f"  ✓ AGE: r={r_age:.3f}, p={p_age:.3e}")
        sig_markers.append('Age')
    if p_sex < 0.05:
        print(f"  ✓ SEX: p={p_sex:.3e}")
        sig_markers.append('Sex')
    if p_ct < 0.05:
        print(f"  ✓ CELL TYPE: p={p_ct:.3e}, η²={eta2_ct:.3f} ({eta2_ct*100:.1f}% variance)")
        sig_markers.append('CellType')
    if p_reg < 0.05:
        print(f"  ✓ REGION: p={p_reg:.3e}, η²={eta2_reg:.3f} ({eta2_reg*100:.1f}% variance)")
        sig_markers.append('Region')
    
    if not sig_markers:
        print(f"  → Technical/Other")
    
    results.append({
        'Factor': factor,
        'age_r': r_age,
        'age_p': p_age,
        'sex_p': p_sex,
        'celltype_p': p_ct,
        'celltype_eta2': eta2_ct,
        'region_p': p_reg,
        'region_eta2': eta2_reg,
        'primary_source': sig_markers[0] if sig_markers else 'Technical'
    })

# Create summary dataframe
results_df = pd.DataFrame(results)
results_df.to_csv('fa_sources_of_variation.csv', index=False)
print("\n" + "="*70)
print("Saved: fa_sources_of_variation.csv")

# Summary stats
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal factors analyzed: {len(results_df)}")

# Age effects
age_factors = results_df[results_df['age_p'] < 0.05]
print(f"Factors with age effect: {len(age_factors)}")

# Cell type effects
ct_factors = results_df[results_df['celltype_p'] < 0.05]
print(f"Factors with cell type effect: {len(ct_factors)}")

# Region effects
reg_factors = results_df[results_df['region_p'] < 0.05]
print(f"Factors with region effect: {len(reg_factors)}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Early vs late factors
early_factors = results_df.head(10)
late_factors = results_df.tail(10)

print(f"\nEarly Factors (1-10):")
print(f"  Mean Cell Type η²: {early_factors['celltype_eta2'].mean():.3f}")
print(f"  Mean Region η²: {early_factors['region_eta2'].mean():.3f}")
print(f"  Mean |Age r|: {np.abs(early_factors['age_r']).mean():.3f}")

print(f"\nLate Factors (41-50):")
print(f"  Mean Cell Type η²: {late_factors['celltype_eta2'].mean():.3f}")
print(f"  Mean Region η²: {late_factors['region_eta2'].mean():.3f}")
print(f"  Mean |Age r|: {np.abs(late_factors['age_r']).mean():.3f}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

