#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the PCA results
pca_df = pd.read_csv('astrocytes_pseudobulk_pca.csv')

print("=== PC correlations with all metadata ===\n")

# Correlation with continuous variable (age)
for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
    age_corr = np.corrcoef(pca_df[pc], pca_df['age'])[0, 1]
    print(f"{pc} vs Age: r = {age_corr:.3f}")

print("\n=== PC associations with categorical variables ===\n")

# Check if PCs differ by region
for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
    print(f"\n{pc} by Region:")
    region_means = pca_df.groupby('region')[pc].mean().sort_values(ascending=False)
    print(region_means)

print("\n=== PC associations with Sex ===\n")
for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
    print(f"\n{pc} by Sex:")
    sex_means = pca_df.groupby('sex')[pc].mean()
    print(sex_means)

# Create more detailed correlation plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PC3 vs Age (highest correlation)
axes[0, 0].scatter(pca_df['age'], pca_df['PC3'], alpha=0.5, s=30)
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('PC3')
axes[0, 0].set_title(f'PC3 vs Age (r = {np.corrcoef(pca_df["PC3"], pca_df["age"])[0,1]:.3f})')

# PC4 vs Age
axes[0, 1].scatter(pca_df['age'], pca_df['PC4'], alpha=0.5, s=30)
axes[0, 1].set_xlabel('Age (years)')
axes[0, 1].set_ylabel('PC4')
axes[0, 1].set_title(f'PC4 vs Age (r = {np.corrcoef(pca_df["PC4"], pca_df["age"])[0,1]:.3f})')

# PC1 by region (boxplot)
region_order = pca_df.groupby('region')['PC1'].mean().sort_values().index
pca_df['region_cat'] = pd.Categorical(pca_df['region'], categories=region_order, ordered=True)
axes[1, 0].boxplot([pca_df[pca_df['region'] == r]['PC1'].values for r in region_order],
                    labels=region_order)
axes[1, 0].set_xlabel('Region')
axes[1, 0].set_ylabel('PC1')
axes[1, 0].set_title('PC1 by Region')
axes[1, 0].tick_params(axis='x', rotation=45)

# PC2 by region (boxplot)
axes[1, 1].boxplot([pca_df[pca_df['region'] == r]['PC2'].values for r in region_order],
                    labels=region_order)
axes[1, 1].set_xlabel('Region')
axes[1, 1].set_ylabel('PC2')
axes[1, 1].set_title('PC2 by Region')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('pca_detailed_exploration.png', dpi=300, bbox_inches='tight')
print("\nSaved: pca_detailed_exploration.png")

# Check variance by animal
print("\n=== Checking animal effects ===")
print(f"Number of unique animals: {pca_df['animal_id'].nunique()}")
print(f"Animals per region: {pca_df.groupby('region')['animal_id'].nunique().to_dict()}")

