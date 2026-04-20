#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the 50 PC sources data
sources_df = pd.read_csv('pc_sources_of_variation_50.csv')

print(f"Visualizing {len(sources_df)} PCs...")

# ========== VISUALIZATION 1: Heatmap of Effect Sizes ==========
print("\nCreating heatmap...")

fig, ax = plt.subplots(figsize=(16, 8))

# Prepare data for heatmap
heatmap_data = sources_df[['celltype_eta2', 'region_eta2', 'age_r', 'sex_p']].copy()
heatmap_data['age_r2'] = heatmap_data['age_r'] ** 2  # Convert to r²
heatmap_data['sex_sig'] = (heatmap_data['sex_p'] < 0.05).astype(float) * 0.05  # Binary indicator
heatmap_data = heatmap_data[['celltype_eta2', 'region_eta2', 'age_r2', 'sex_sig']].T

heatmap_data.columns = [f'PC{i+1}' for i in range(len(sources_df))]

sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.2f',
            cbar_kws={'label': 'Effect Size'}, ax=ax, vmin=0, vmax=1)

ax.set_yticklabels(['Cell Type (η²)', 'Region (η²)', 'Age (r²)', 'Sex (sig)'], rotation=0)
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_title('Sources of Variation Across 50 PCs', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('pca50_sources_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: pca50_sources_heatmap.png")

# ========== VISUALIZATION 2: Stacked Bar Chart - Variance Decomposition ==========
print("\nCreating variance decomposition stacked bar chart...")

fig, ax = plt.subplots(figsize=(18, 6))

pcs = sources_df['PC'].str.replace('PC', '').astype(int).values

# Variance components
celltype_var = sources_df['celltype_eta2'].values
region_var = sources_df['region_eta2'].values
age_var = sources_df['age_r'].apply(lambda x: x**2).values
sex_var = sources_df['sex_p'].apply(lambda p: 0.01 if p < 0.05 else 0).values

# Other variance = remainder
total_var = celltype_var + region_var + age_var + sex_var
other_var = np.maximum(0, 1 - total_var)

# Stacked bars
ax.bar(pcs, celltype_var, label='Cell Type', color='#e74c3c', alpha=0.8)
ax.bar(pcs, region_var, bottom=celltype_var, label='Region', color='#3498db', alpha=0.8)
ax.bar(pcs, age_var, bottom=celltype_var+region_var, label='Age', color='#2ecc71', alpha=0.8)
ax.bar(pcs, sex_var, bottom=celltype_var+region_var+age_var, label='Sex', color='#9b59b6', alpha=0.8)
ax.bar(pcs, other_var, bottom=celltype_var+region_var+age_var+sex_var, 
       label='Other/Technical', color='#95a5a6', alpha=0.5)

ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Proportion of Variance Explained', fontsize=12, fontweight='bold')
ax.set_title('Variance Decomposition Across PCs\n(Stacked bars show relative contribution)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xticks(pcs)
ax.set_xticklabels([f'PC{i}' for i in pcs], rotation=90, fontsize=7)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('pca50_variance_decomposition.png', dpi=300, bbox_inches='tight')
print("Saved: pca50_variance_decomposition.png")

# ========== VISUALIZATION 3: Scatter - Cell Type vs Region ==========
print("\nCreating cell type vs region scatter...")

fig, ax = plt.subplots(figsize=(10, 8))

# Color by whether age is significant
colors = ['red' if p < 0.05 else 'gray' for p in sources_df['age_p']]
sizes = [abs(r)*500 for r in sources_df['age_r']]

scatter = ax.scatter(sources_df['celltype_eta2'], 
                     sources_df['region_eta2'],
                     c=colors,
                     s=sizes,
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1.5)

# Add PC labels for notable ones
for idx, row in sources_df.iterrows():
    pc_num = int(row['PC'].replace('PC', ''))
    # Label if age significant or high cell type/region
    if row['age_p'] < 0.05 or row['celltype_eta2'] > 0.5 or row['region_eta2'] > 0.5:
        ax.annotate(f'{pc_num}', 
                    (row['celltype_eta2'], row['region_eta2']),
                    fontsize=8,
                    ha='center',
                    va='center',
                    fontweight='bold')

ax.set_xlabel('Cell Type Effect (η²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Region Effect (η²)', fontsize=12, fontweight='bold')
ax.set_title('Cell Type vs Region Contribution to Each PC\n(Size = age correlation strength, Red = age p<0.05)', 
             fontsize=13, fontweight='bold')

# Add quadrant lines
ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca50_celltype_vs_region.png', dpi=300, bbox_inches='tight')
print("Saved: pca50_celltype_vs_region.png")

# ========== VISUALIZATION 4: Line Plots - Trends Across PCs ==========
print("\nCreating trend line plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cell Type
ax = axes[0, 0]
ax.plot(pcs, sources_df['celltype_eta2'], 'o-', color='#e74c3c', linewidth=2, markersize=4)
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('Cell Type η²', fontweight='bold')
ax.set_title('Cell Type Effects Across PCs', fontweight='bold')
ax.grid(alpha=0.3)

# Region
ax = axes[0, 1]
ax.plot(pcs, sources_df['region_eta2'], 'o-', color='#3498db', linewidth=2, markersize=4)
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('Region η²', fontweight='bold')
ax.set_title('Region Effects Across PCs', fontweight='bold')
ax.grid(alpha=0.3)

# Age
ax = axes[1, 0]
ax.plot(pcs, np.abs(sources_df['age_r']), 'o-', color='#2ecc71', linewidth=2, markersize=4)
ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='|r|=0.3')
ax.set_xlabel('PC', fontweight='bold')
ax.set_ylabel('|Age Correlation|', fontweight='bold')
ax.set_title('Age Effects Across PCs', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Summary statistics
ax = axes[1, 1]
summary_data = {
    'Cell Type\n(mean η²)': sources_df['celltype_eta2'].mean(),
    'Region\n(mean η²)': sources_df['region_eta2'].mean(),
    'Age\n(mean |r|)': np.abs(sources_df['age_r']).mean(),
    'Age sig\n(count)': (sources_df['age_p'] < 0.05).sum()
}

bars = ax.bar(range(len(summary_data)), list(summary_data.values()), 
              color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'], alpha=0.7)
ax.set_xticks(range(len(summary_data)))
ax.set_xticklabels(list(summary_data.keys()))
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Summary Statistics', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, summary_data.values())):
    if i < 3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('pca50_trends.png', dpi=300, bbox_inches='tight')
print("Saved: pca50_trends.png")

# ========== Summary Statistics ==========
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\n1. CELL TYPE DOMINANCE:")
ct_dominant = sources_df[sources_df['celltype_eta2'] > 0.5]
print(f"   PCs dominated by cell type (>50%): {len(ct_dominant)}")
if len(ct_dominant) > 0:
    print(f"   PCs: {', '.join(ct_dominant['PC'].tolist())}")

print("\n2. REGION DOMINANCE:")
reg_dominant = sources_df[sources_df['region_eta2'] > 0.3]
print(f"   PCs dominated by region (>30%): {len(reg_dominant)}")
if len(reg_dominant) > 0:
    print(f"   PCs: {', '.join(reg_dominant['PC'].tolist()[:10])}...")  # Show first 10

print("\n3. AGE-ASSOCIATED (p<0.05):")
age_sig = sources_df[sources_df['age_p'] < 0.05]
print(f"   PCs with age effects: {len(age_sig)}")
if len(age_sig) > 0:
    print(f"   PCs: {', '.join(age_sig['PC'].tolist()[:10])}...")

print("\n4. 'CLEAN' AGE PCs (age sig, low cell type/region):")
clean_age = sources_df[(sources_df['age_p'] < 0.05) & 
                       (sources_df['celltype_eta2'] < 0.1) & 
                       (sources_df['region_eta2'] < 0.3)]
if not clean_age.empty:
    print(f"   Clean age PCs: {', '.join(clean_age['PC'].tolist())}")
    for idx, row in clean_age.iterrows():
        print(f"      {row['PC']}: age r={row['age_r']:.3f}, cell type η²={row['celltype_eta2']:.3f}, region η²={row['region_eta2']:.3f}")
else:
    print("   No perfectly 'clean' age PCs")

print("\n" + "="*70)
print("Complete!")
print("="*70)

