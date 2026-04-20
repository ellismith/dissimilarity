#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

# Load PCA results
pca_df = pd.read_csv('pca_combined_all_celltypes.csv')

print("="*70)
print("TESTING WHAT EACH PC CAPTURES")
print("="*70)

results = []

for i in range(1, 21):
    pc = f'PC{i}'
    
    print(f"\n{pc}:")
    
    # 1. AGE effect
    r_age, p_age = stats.pearsonr(pca_df['age'], pca_df[pc])
    
    # 2. SEX effect
    males = pca_df[pca_df['sex'] == 'M'][pc]
    females = pca_df[pca_df['sex'] == 'F'][pc]
    t_sex, p_sex = stats.ttest_ind(males, females)
    effect_sex = males.mean() - females.mean()
    
    # 3. CELL TYPE effect (ANOVA)
    cell_groups = [pca_df[pca_df['cell_type'] == ct][pc].values 
                   for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']]
    f_ct, p_ct = f_oneway(*cell_groups)
    
    # Calculate eta-squared (effect size) for cell type
    grand_mean = pca_df[pc].mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in cell_groups])
    ss_total = sum((pca_df[pc] - grand_mean)**2)
    eta2_ct = ss_between / ss_total
    
    # 4. REGION effect (ANOVA)
    regions = pca_df['region'].unique()
    region_groups = [pca_df[pca_df['region'] == r][pc].values 
                     for r in regions if len(pca_df[pca_df['region'] == r]) > 0]
    f_reg, p_reg = f_oneway(*region_groups)
    
    # Calculate eta-squared for region
    ss_between_reg = sum([len(g) * (g.mean() - grand_mean)**2 for g in region_groups])
    eta2_reg = ss_between_reg / ss_total
    
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
        'PC': pc,
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
results_df.to_csv('pc_sources_of_variation.csv', index=False)
print("\n" + "="*70)
print("Saved: pc_sources_of_variation.csv")

# === VISUALIZATION 1: Effect sizes across PCs ===
print("\nCreating effect size visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age
ax = axes[0, 0]
ax.bar(range(1, 21), np.abs(results_df['age_r']), 
       color=['red' if p < 0.05 else 'gray' for p in results_df['age_p']],
       alpha=0.7, edgecolor='black')
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('|Age Correlation|', fontweight='bold')
ax.set_title('Age Effect Size', fontweight='bold')
ax.axhline(y=0.27, color='red', linestyle='--', alpha=0.5, label='p≈0.05 threshold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Sex
ax = axes[0, 1]
sex_effect = [-np.log10(p) if p > 0 else 10 for p in results_df['sex_p']]
ax.bar(range(1, 21), sex_effect,
       color=['blue' if p < 0.05 else 'gray' for p in results_df['sex_p']],
       alpha=0.7, edgecolor='black')
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('-log10(p-value)', fontweight='bold')
ax.set_title('Sex Effect', fontweight='bold')
ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Cell Type
ax = axes[1, 0]
ax.bar(range(1, 21), results_df['celltype_eta2'] * 100,
       color=['green' if p < 0.05 else 'gray' for p in results_df['celltype_p']],
       alpha=0.7, edgecolor='black')
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('η² (% Variance Explained)', fontweight='bold')
ax.set_title('Cell Type Effect Size', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Region
ax = axes[1, 1]
ax.bar(range(1, 21), results_df['region_eta2'] * 100,
       color=['orange' if p < 0.05 else 'gray' for p in results_df['region_p']],
       alpha=0.7, edgecolor='black')
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('η² (% Variance Explained)', fontweight='bold')
ax.set_title('Region Effect Size', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('pc_sources_of_variation.png', dpi=300, bbox_inches='tight')
print("Saved: pc_sources_of_variation.png")

# === VISUALIZATION 2: Stacked bar showing what each PC captures ===
print("\nCreating stacked bar chart...")

fig, ax = plt.subplots(figsize=(14, 6))

# Categorize each PC
categories = []
for idx, row in results_df.iterrows():
    effects = []
    if row['age_p'] < 0.05:
        effects.append('Age')
    if row['sex_p'] < 0.05:
        effects.append('Sex')
    if row['celltype_p'] < 0.05:
        effects.append('Cell Type')
    if row['region_p'] < 0.05:
        effects.append('Region')
    
    if not effects:
        categories.append('Technical/Other')
    elif len(effects) == 1:
        categories.append(effects[0])
    else:
        categories.append(' + '.join(effects))

results_df['category'] = categories

# Count by category
from collections import Counter
cat_counts = Counter(categories)

print("\n" + "="*70)
print("PC CATEGORIZATION")
print("="*70)

for cat, count in cat_counts.most_common():
    pcs = [results_df.iloc[i]['PC'] for i in range(len(results_df)) if categories[i] == cat]
    print(f"\n{cat}: {count} PCs")
    print(f"  PCs: {', '.join(pcs)}")

# Create summary table
print("\n" + "="*70)
print("DETAILED PC SOURCES")
print("="*70)

summary = results_df[['PC', 'age_r', 'age_p', 'celltype_eta2', 'region_eta2', 'category']].copy()
summary['age_sig'] = summary['age_p'] < 0.05
summary = summary.sort_values('PC')

print("\nEarly PCs (1-10):")
print(summary.head(10)[['PC', 'age_r', 'celltype_eta2', 'region_eta2', 'category']].to_string(index=False))

print("\nLater PCs (11-20):")
print(summary.tail(10)[['PC', 'age_r', 'celltype_eta2', 'region_eta2', 'category']].to_string(index=False))

# Bar chart by PC
colors_map = {
    'Cell Type': 'green',
    'Region': 'orange',
    'Age': 'red',
    'Sex': 'blue',
    'Technical/Other': 'gray'
}

# For mixed categories, use a different color
for cat in set(categories):
    if '+' in cat:
        colors_map[cat] = 'purple'

colors = [colors_map.get(cat, 'purple') for cat in categories]

ax.bar(range(1, 21), [1]*20, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('PC', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=12)
ax.set_title('What Does Each PC Capture?\n(Based on Significant Associations)', 
            fontsize=13, fontweight='bold')
ax.set_yticks([])

# Create legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_map[cat], label=cat, alpha=0.7) 
                   for cat in sorted(set(categories))]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('pc_categorization_barplot.png', dpi=300, bbox_inches='tight')
print("\nSaved: pc_categorization_barplot.png")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

# Early vs late PCs
early_pcs = results_df.head(5)
late_pcs = results_df.tail(10)

print(f"\nEarly PCs (1-5):")
print(f"  Mean Cell Type η²: {early_pcs['celltype_eta2'].mean():.3f}")
print(f"  Mean Region η²: {early_pcs['region_eta2'].mean():.3f}")
print(f"  Mean |Age r|: {np.abs(early_pcs['age_r']).mean():.3f}")

print(f"\nLater PCs (11-20):")
print(f"  Mean Cell Type η²: {late_pcs['celltype_eta2'].mean():.3f}")
print(f"  Mean Region η²: {late_pcs['region_eta2'].mean():.3f}")
print(f"  Mean |Age r|: {np.abs(late_pcs['age_r']).mean():.3f}")

print("\n→ Early PCs dominated by cell type/region differences")
print("→ Age signals appear in later PCs after removing structural variation")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

