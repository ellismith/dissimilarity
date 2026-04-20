#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sources of variation data (assuming it has 50 PCs)
sources_df = pd.read_csv('pc_sources_of_variation.csv')

# Filter to first 50 PCs if more exist
sources_df = sources_df[sources_df['PC'].str.replace('PC', '').astype(int) <= 50]

print(f"Creating variance decomposition for {len(sources_df)} PCs...")

# ========== VISUALIZATION: Stacked Bar Chart ==========
fig, ax = plt.subplots(figsize=(16, 6))

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
ax.set_xticklabels([f'PC{i}' for i in pcs], rotation=90, fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('pc_variance_decomposition_50.png', dpi=300, bbox_inches='tight')
print("Saved: pc_variance_decomposition_50.png")

