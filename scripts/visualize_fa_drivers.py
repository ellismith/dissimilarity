#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the FA sources data
sources_df = pd.read_csv('fa_sources_of_variation.csv')

print("Creating visualizations of FA drivers...")

# ========== VISUALIZATION 1: Heatmap of effect sizes ==========
fig, ax = plt.subplots(figsize=(16, 8))

# Create matrix: Factors × sources
data_matrix = sources_df[['Factor', 'celltype_eta2', 'region_eta2', 'age_r', 'sex_p']].copy()
data_matrix['age_r2'] = data_matrix['age_r'] ** 2  # Convert to variance explained
data_matrix['sex_sig'] = (data_matrix['sex_p'] < 0.05).astype(float) * 0.02  # Binary indicator

# Create heatmap data
heatmap_data = data_matrix[['celltype_eta2', 'region_eta2', 'age_r2', 'sex_sig']].values
factor_labels = [f"F{i}" for i in range(1, 51)]

# Plot
sns.heatmap(heatmap_data.T, 
            xticklabels=factor_labels,
            yticklabels=['Cell Type (η²)', 'Region (η²)', 'Age (r²)', 'Sex (sig)'],
            cmap='YlOrRd',
            vmin=0,
            vmax=0.9,
            cbar_kws={'label': 'Variance Explained (η²) or r²'},
            linewidths=0.5,
            ax=ax)

ax.set_title('What Drives Each Factor?\n(Variance explained by different sources)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Source of Variation', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fa_drivers_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: fa_drivers_heatmap.png")

# ========== VISUALIZATION 2: Stacked bar chart (subset to show pattern) ==========
fig, ax = plt.subplots(figsize=(16, 6))

# Show every 2nd factor to avoid crowding
factors_to_plot = range(0, 50, 2)
factor_nums = [i+1 for i in factors_to_plot]

# Prepare data
celltype_var = sources_df.iloc[factors_to_plot]['celltype_eta2'].values
region_var = sources_df.iloc[factors_to_plot]['region_eta2'].values
age_var = sources_df.iloc[factors_to_plot]['age_r'].apply(lambda x: x**2).values
sex_var = sources_df.iloc[factors_to_plot]['sex_p'].apply(lambda p: 0.01 if p < 0.05 else 0).values

# Other variance
total_var = celltype_var + region_var + age_var + sex_var
other_var = np.maximum(0, 1 - total_var)

# Stacked bars
x_pos = range(len(factor_nums))
ax.bar(x_pos, celltype_var, label='Cell Type', color='#e74c3c', alpha=0.8)
ax.bar(x_pos, region_var, bottom=celltype_var, label='Region', color='#3498db', alpha=0.8)
ax.bar(x_pos, age_var, bottom=celltype_var+region_var, label='Age', color='#2ecc71', alpha=0.8)
ax.bar(x_pos, sex_var, bottom=celltype_var+region_var+age_var, label='Sex', color='#9b59b6', alpha=0.8)
ax.bar(x_pos, other_var, bottom=celltype_var+region_var+age_var+sex_var, 
       label='Other/Technical', color='#95a5a6', alpha=0.5)

ax.set_xlabel('Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Proportion of Variance Explained', fontsize=12, fontweight='bold')
ax.set_title('Variance Decomposition Across Factors (showing every 2nd factor)\n(Stacked bars show relative contribution)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in factor_nums], rotation=45)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('fa_variance_decomposition.png', dpi=300, bbox_inches='tight')
print("Saved: fa_variance_decomposition.png")

# ========== VISUALIZATION 3: Scatter plot - Cell Type vs Region ==========
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
                     linewidth=1)

# Add factor labels for notable ones
for idx, row in sources_df.iterrows():
    factor_num = int(row['Factor'].replace('Factor', ''))
    # Label if age significant or high cell type/region
    if row['age_p'] < 0.05 or row['celltype_eta2'] > 0.3 or row['region_eta2'] > 0.3:
        ax.annotate(f'{factor_num}', 
                    (row['celltype_eta2'], row['region_eta2']),
                    fontsize=7,
                    ha='center',
                    va='center',
                    fontweight='bold')

ax.set_xlabel('Cell Type Effect (η²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Region Effect (η²)', fontsize=12, fontweight='bold')
ax.set_title('Cell Type vs Region Contribution to Each Factor\n(Size = age correlation strength, Red = age significant)', 
             fontsize=13, fontweight='bold')

# Add quadrant lines
ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.3)

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fa_celltype_vs_region.png', dpi=300, bbox_inches='tight')
print("Saved: fa_celltype_vs_region.png")

# ========== VISUALIZATION 4: Comparison with PCA ==========
try:
    pca_sources = pd.read_csv('pc_sources_of_variation.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cell Type comparison
    ax = axes[0, 0]
    ax.plot(range(1, 21), pca_sources['celltype_eta2'], 'o-', 
            color='blue', linewidth=2, label='PCA', markersize=6)
    ax.plot(range(1, 51), sources_df['celltype_eta2'], 's-', 
            color='red', linewidth=1, label='FA', markersize=4, alpha=0.7)
    ax.set_xlabel('Component/Factor', fontweight='bold')
    ax.set_ylabel('Cell Type η²', fontweight='bold')
    ax.set_title('Cell Type Effects: PCA vs FA', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Region comparison
    ax = axes[0, 1]
    ax.plot(range(1, 21), pca_sources['region_eta2'], 'o-', 
            color='blue', linewidth=2, label='PCA', markersize=6)
    ax.plot(range(1, 51), sources_df['region_eta2'], 's-', 
            color='red', linewidth=1, label='FA', markersize=4, alpha=0.7)
    ax.set_xlabel('Component/Factor', fontweight='bold')
    ax.set_ylabel('Region η²', fontweight='bold')
    ax.set_title('Region Effects: PCA vs FA', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Age comparison
    ax = axes[1, 0]
    ax.plot(range(1, 21), np.abs(pca_sources['age_r']), 'o-', 
            color='blue', linewidth=2, label='PCA', markersize=6)
    ax.plot(range(1, 51), np.abs(sources_df['age_r']), 's-', 
            color='red', linewidth=1, label='FA', markersize=4, alpha=0.7)
    ax.set_xlabel('Component/Factor', fontweight='bold')
    ax.set_ylabel('|Age Correlation|', fontweight='bold')
    ax.set_title('Age Effects: PCA vs FA', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    
    categories = ['Cell Type\n(mean η²)', 'Region\n(mean η²)', 'Age\n(mean |r|)']
    pca_vals = [pca_sources['celltype_eta2'].mean(), 
                pca_sources['region_eta2'].mean(),
                np.abs(pca_sources['age_r']).mean()]
    fa_vals = [sources_df['celltype_eta2'].mean(),
               sources_df['region_eta2'].mean(),
               np.abs(sources_df['age_r']).mean()]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, pca_vals, width, label='PCA (20)', color='blue', alpha=0.7)
    ax.bar(x + width/2, fa_vals, width, label='FA (50)', color='red', alpha=0.7)
    
    ax.set_ylabel('Mean Effect Size', fontweight='bold')
    ax.set_title('Average Effects: PCA vs FA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_vs_fa_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: pca_vs_fa_comparison.png")
    
except Exception as e:
    print(f"Could not create PCA comparison: {e}")

print("\n" + "="*70)
print("Complete!")
print("="*70)

