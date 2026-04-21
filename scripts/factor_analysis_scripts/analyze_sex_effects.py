#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the factor analysis results for each region/cell type
analyses = [
    ('GABAergic', 'dlPFC', 10, 'factor_analysis_GABAergic_dlPFC.csv'),
    ('GABAergic', 'ACC', 7, 'factor_analysis_GABAergic_ACC.csv'),
    ('GABAergic', 'M1', 3, 'factor_analysis_GABAergic_M1.csv'),
    ('Astrocytes', 'CN', 7, 'factor_analysis_CN.csv'),
    ('Astrocytes', 'M1', 8, 'factor_analysis_M1.csv'),
    ('Astrocytes', 'ACC', 6, 'factor_analysis_ACC.csv')
]

results = []

print("="*70)
print("SEX EFFECTS ON AGE-ASSOCIATED FACTORS")
print("="*70)

for cell_type, region, factor_num, file in analyses:
    try:
        df = pd.read_csv(file)
        factor_col = f'Factor{factor_num}'
        
        # Calculate age correlation
        age_corr, age_pval = stats.pearsonr(df[factor_col], df['age'])
        
        # Calculate sex difference (t-test)
        male_vals = df[df['sex'] == 'M'][factor_col]
        female_vals = df[df['sex'] == 'F'][factor_col]
        t_stat, sex_pval = stats.ttest_ind(male_vals, female_vals)
        
        # Calculate effect sizes
        male_mean = male_vals.mean()
        female_mean = female_vals.mean()
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        cohens_d = (male_mean - female_mean) / pooled_std
        
        # Partial correlation: age controlling for sex
        # Use residuals from sex regression
        from sklearn.linear_model import LinearRegression
        X_sex = pd.get_dummies(df['sex'], drop_first=True).values.reshape(-1, 1)
        
        # Regress factor on sex
        lr_factor = LinearRegression().fit(X_sex, df[factor_col])
        factor_resid = df[factor_col] - lr_factor.predict(X_sex)
        
        # Regress age on sex
        lr_age = LinearRegression().fit(X_sex, df['age'])
        age_resid = df['age'] - lr_age.predict(X_sex)
        
        # Correlation of residuals = partial correlation
        partial_corr, partial_pval = stats.pearsonr(factor_resid, age_resid)
        
        print(f"\n{cell_type} - {region} - Factor{factor_num}")
        print(f"  Age correlation: r = {age_corr:.3f}, p = {age_pval:.3e}")
        print(f"  Sex difference:  t = {t_stat:.3f}, p = {sex_pval:.3e}, d = {cohens_d:.3f}")
        print(f"    Male mean:   {male_mean:.3f}")
        print(f"    Female mean: {female_mean:.3f}")
        print(f"  Age|Sex (partial): r = {partial_corr:.3f}, p = {partial_pval:.3e}")
        
        if sex_pval < 0.05:
            print(f"  *** SIGNIFICANT SEX EFFECT ***")
        
        results.append({
            'cell_type': cell_type,
            'region': region,
            'factor': factor_num,
            'age_corr': age_corr,
            'age_pval': age_pval,
            'sex_t': t_stat,
            'sex_pval': sex_pval,
            'cohens_d': cohens_d,
            'male_mean': male_mean,
            'female_mean': female_mean,
            'partial_corr': partial_corr,
            'partial_pval': partial_pval,
            'sex_significant': sex_pval < 0.05
        })
        
    except Exception as e:
        print(f"\nError with {cell_type} {region}: {e}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv('sex_effects_summary.csv', index=False)
print("\n" + "="*70)
print("Saved: sex_effects_summary.csv")

# === Visualizations ===
print("\nCreating visualizations...")

# 1. Compare age correlation vs partial correlation (age controlling for sex)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Age correlation vs Sex effect
ax = axes[0]
colors = ['steelblue' if not sig else 'red' for sig in results_df['sex_significant']]
labels = [f"{row['cell_type'][:4]}\n{row['region']}" for _, row in results_df.iterrows()]

x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['age_corr'].abs(), width, 
              label='|Age correlation|', alpha=0.7, color='steelblue')
bars2 = ax.bar(x + width/2, results_df['partial_corr'].abs(), width,
              label='|Age|Sex (partial)|', alpha=0.7, color='orange')

ax.set_ylabel('|Correlation|', fontsize=11, fontweight='bold')
ax.set_title('Age Correlation: Raw vs. Controlling for Sex', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Sex effect sizes (Cohen's d)
ax = axes[1]
colors = ['red' if sig else 'gray' for sig in results_df['sex_significant']]
bars = ax.barh(labels, results_df['cohens_d'], color=colors, alpha=0.7)

ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel("Cohen's d (Male - Female)", fontsize=11, fontweight='bold')
ax.set_title('Sex Differences in Age-Associated Factors', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
                  Patch(facecolor='gray', alpha=0.7, label='Not significant')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('sex_effects_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: sex_effects_comparison.png")

# 2. Scatter plots: Factor vs Age, colored by Sex
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(results_df.iterrows()):
    ax = axes[idx]
    
    # Load data for this analysis
    cell_type = row['cell_type']
    region = row['region']
    factor_num = row['factor']
    
    # Find the file
    if cell_type == 'GABAergic':
        file = f'factor_analysis_GABAergic_{region}.csv'
    else:
        file = f'factor_analysis_{region}.csv'
    
    try:
        df = pd.read_csv(file)
        factor_col = f'Factor{factor_num}'
        
        # Plot
        for sex, color, marker in [('M', 'blue', 'o'), ('F', 'red', '^')]:
            mask = df['sex'] == sex
            ax.scatter(df.loc[mask, 'age'], df.loc[mask, factor_col],
                      c=color, marker=marker, s=40, alpha=0.6, label=sex)
        
        # Add regression lines
        for sex, color in [('M', 'blue'), ('F', 'red')]:
            mask = df['sex'] == sex
            z = np.polyfit(df.loc[mask, 'age'], df.loc[mask, factor_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Age (years)', fontsize=9)
        ax.set_ylabel(f'Factor{factor_num}', fontsize=9)
        ax.set_title(f"{cell_type} - {region}\n" + 
                    f"Age: r={row['age_corr']:.2f}, " +
                    f"Sex: p={row['sex_pval']:.3f}",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center',
               transform=ax.transAxes)

plt.tight_layout()
plt.savefig('factor_age_by_sex.png', dpi=300, bbox_inches='tight')
print("Saved: factor_age_by_sex.png")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nFactors with significant sex effects (p<0.05): {results_df['sex_significant'].sum()}/{len(results_df)}")
print("\nFactors ranked by |sex effect size| (Cohen's d):")
print(results_df[['cell_type', 'region', 'factor', 'cohens_d', 'sex_pval']]
      .sort_values('cohens_d', key=abs, ascending=False)
      .to_string(index=False))

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

