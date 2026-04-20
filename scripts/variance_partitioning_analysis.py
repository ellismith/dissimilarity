#!/usr/bin/env python
"""
Variance partitioning: How much variance in PC scores is explained by age, sex, region, cell type?
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import glob
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("VARIANCE PARTITIONING ANALYSIS")
print("="*70)

# Get all PCA files
pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')
print(f"\nFound {len(pca_files)} PCA files")

results = []

for pca_file in pca_files:
    basename = pca_file.split('/')[-1]
    parts = basename.replace('pca_analysis_', '').replace('.csv', '').split('_')
    
    if len(parts) < 2:
        continue
    
    region = parts[-1]
    cell_type = '_'.join(parts[:-1])
    
    try:
        df = pd.read_csv(pca_file)
        
        if df.empty:
            continue
        
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        
        if len(pc_cols) == 0:
            continue
        
        # Encode sex as numeric
        if 'sex' in df.columns:
            sex_encoded = pd.get_dummies(df['sex'], drop_first=True)
            if len(sex_encoded.columns) > 0:
                df['sex_numeric'] = sex_encoded.iloc[:, 0]
            else:
                df['sex_numeric'] = 0
        else:
            df['sex_numeric'] = 0
        
        # For each PC, calculate variance explained by age and sex
        for pc in pc_cols:
            # Total variance
            total_var = df[pc].var()
            
            # Variance explained by age alone (R²)
            age_r2 = stats.spearmanr(df['age'], df[pc])[0] ** 2
            
            # Variance explained by sex alone
            if df['sex_numeric'].nunique() > 1:
                sex_r2 = stats.spearmanr(df['sex_numeric'], df[pc])[0] ** 2
            else:
                sex_r2 = 0
            
            # Combined model (age + sex)
            X = df[['age', 'sex_numeric']].values
            y = df[pc].values
            
            lr = LinearRegression()
            lr.fit(X, y)
            combined_r2 = lr.score(X, y)
            
            # Residual variance
            residual_r2 = 1 - combined_r2
            
            results.append({
                'cell_type': cell_type,
                'region': region,
                'PC': pc,
                'age_r2': age_r2,
                'sex_r2': sex_r2,
                'combined_r2': combined_r2,
                'residual_r2': residual_r2,
                'n_animals': len(df)
            })
    
    except Exception as e:
        print(f"ERROR: {basename} - {e}")
        continue

# Convert to dataframe
results_df = pd.DataFrame(results)

# Save
results_df.to_csv('/scratch/easmit31/factor_analysis/variance_partitioning_celltype_level.csv', index=False)
print(f"\nSaved: /scratch/easmit31/factor_analysis/variance_partitioning_celltype_level.csv")

output_dir = '/scratch/easmit31/factor_analysis/subtype_variability/figures'

# Figure 1: Simple stacked bar chart
print("\nCreating Figure 1: Average variance partitioning")

avg_age = results_df['age_r2'].mean()
avg_sex = results_df['sex_r2'].mean()
avg_residual = results_df['residual_r2'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['All PC Scores']
age_vals = [avg_age]
sex_vals = [avg_sex]
residual_vals = [avg_residual]

# Stacked bar
p1 = ax.barh(categories, age_vals, color='#e74c3c', label='Age')
p2 = ax.barh(categories, sex_vals, left=age_vals, color='#3498db', label='Sex')
p3 = ax.barh(categories, residual_vals, left=[age_vals[0] + sex_vals[0]], 
            color='#95a5a6', label='Residual/Individual')

# Add percentage labels
ax.text(avg_age/2, 0, f'Age\n{avg_age*100:.1f}%',
       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(avg_age + avg_sex/2, 0, f'Sex\n{avg_sex*100:.1f}%',
       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(avg_age + avg_sex + avg_residual/2, 0, f'Residual\n{avg_residual*100:.1f}%',
       ha='center', va='center', fontsize=12, fontweight='bold', color='white')

ax.set_xlabel('Proportion of Variance', fontsize=12, fontweight='bold')
ax.set_title('Average Variance Partitioning Across All PC Scores\n(cell type-region level)',
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 1])
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_partitioning_average.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_partitioning_average.png")
plt.close()

# Figure 2: By cell type
print("\nCreating Figure 2: Variance partitioning by cell type")

cell_type_avg = results_df.groupby('cell_type')[['age_r2', 'sex_r2', 'residual_r2']].mean()

fig, ax = plt.subplots(figsize=(10, 6))
cell_type_avg.plot(kind='barh', stacked=True, ax=ax,
                   color=['#e74c3c', '#3498db', '#95a5a6'])
ax.set_xlabel('Proportion of Variance', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
ax.set_title('Variance Partitioning by Cell Type',
             fontsize=14, fontweight='bold')
ax.legend(['Age', 'Sex', 'Residual/Individual'], loc='lower right')
ax.set_xlim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_partitioning_by_celltype.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_partitioning_by_celltype.png")
plt.close()

# Figure 3: By PC number
print("\nCreating Figure 3: Variance partitioning by PC")

pc_avg = results_df.groupby('PC')[['age_r2', 'sex_r2', 'residual_r2']].mean()
pc_order = [f'PC{i}' for i in range(1, 11)]
pc_avg = pc_avg.reindex(pc_order)

fig, ax = plt.subplots(figsize=(12, 6))
pc_avg.plot(kind='bar', stacked=True, ax=ax,
            color=['#e74c3c', '#3498db', '#95a5a6'])
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Proportion of Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance Partitioning by PC\n(averaged across all cell type-region combinations)',
             fontsize=14, fontweight='bold')
ax.legend(['Age', 'Sex', 'Residual/Individual'], loc='upper right')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_partitioning_by_pc.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_partitioning_by_pc.png")
plt.close()

# Figure 4: Heatmap by cell type and region
print("\nCreating Figure 4: Heatmap of age variance by cell type-region")

age_pivot = results_df.groupby(['cell_type', 'region'])['age_r2'].mean().reset_index()
age_heatmap = age_pivot.pivot(index='cell_type', columns='region', values='age_r2')

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(age_heatmap, annot=True, fmt='.3f', cmap='Reds',
            cbar_kws={'label': 'R² (Age)'}, ax=ax, vmin=0, vmax=0.3)
ax.set_title('Variance Explained by Age\n(averaged across all PCs per cell type-region)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_age_by_celltype_region.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_age_by_celltype_region.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nAverage variance explained:")
print(f"  Age:      {results_df['age_r2'].mean()*100:.2f}%")
print(f"  Sex:      {results_df['sex_r2'].mean()*100:.2f}%")
print(f"  Combined: {results_df['combined_r2'].mean()*100:.2f}%")
print(f"  Residual: {results_df['residual_r2'].mean()*100:.2f}%")

print(f"\nBy cell type:")
for ct in results_df['cell_type'].unique():
    ct_data = results_df[results_df['cell_type'] == ct]
    print(f"  {ct}:")
    print(f"    Age: {ct_data['age_r2'].mean()*100:.2f}%")
    print(f"    Sex: {ct_data['sex_r2'].mean()*100:.2f}%")

print("\n" + "="*70)
print("DONE!")
print("="*70)
