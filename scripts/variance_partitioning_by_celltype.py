#!/usr/bin/env python
"""
Variance partitioning by cell type (averaged across regions)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import glob
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("VARIANCE PARTITIONING BY CELL TYPE")
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
        
        # For each PC
        for pc in pc_cols:
            # Age R²
            age_r2 = stats.spearmanr(df['age'], df[pc])[0] ** 2
            
            # Sex R²
            if df['sex_numeric'].nunique() > 1:
                sex_r2 = stats.spearmanr(df['sex_numeric'], df[pc])[0] ** 2
            else:
                sex_r2 = 0
            
            # Combined model
            X = df[['age', 'sex_numeric']].values
            y = df[pc].values
            
            lr = LinearRegression()
            lr.fit(X, y)
            combined_r2 = lr.score(X, y)
            
            # Residual
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
print(f"\nProcessed {len(results_df)} cell type-region-PC combinations")

output_dir = '/scratch/easmit31/factor_analysis/subtype_variability/figures'

# Figure 1: Stacked bar by cell type and PC (averaged across regions)
print("\nCreating Figure 1: Variance partitioning by cell type and PC")

# Average across regions for each cell type-PC combination
celltype_pc_avg = results_df.groupby(['cell_type', 'PC'])[['age_r2', 'sex_r2', 'residual_r2']].mean().reset_index()

# Get unique cell types and sort PCs
cell_types = ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']
pc_order = [f'PC{i}' for i in range(1, 11)]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, cell_type in enumerate(cell_types):
    ct_data = celltype_pc_avg[celltype_pc_avg['cell_type'] == cell_type]
    
    if len(ct_data) == 0:
        continue
    
    # Reorder by PC
    ct_data = ct_data.set_index('PC').reindex(pc_order).reset_index()
    
    # Create stacked bar
    x = np.arange(len(pc_order))
    width = 0.8
    
    axes[idx].bar(x, ct_data['age_r2'], width, label='Age', color='#e74c3c')
    axes[idx].bar(x, ct_data['sex_r2'], width, bottom=ct_data['age_r2'], 
                  label='Sex', color='#3498db')
    axes[idx].bar(x, ct_data['residual_r2'], width, 
                  bottom=ct_data['age_r2'] + ct_data['sex_r2'],
                  label='Residual/Individual', color='#95a5a6')
    
    axes[idx].set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Proportion of Variance', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{cell_type}\n(averaged across regions)', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(pc_order, rotation=0)
    axes[idx].set_ylim([0, 1])
    axes[idx].legend(loc='upper right', fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_partitioning_by_celltype_and_pc.png', 
            dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_partitioning_by_celltype_and_pc.png")
plt.close()

# Figure 2: Overall by cell type (averaged across all PCs and regions)
print("\nCreating Figure 2: Overall variance partitioning by cell type")

celltype_overall = results_df.groupby('cell_type')[['age_r2', 'sex_r2', 'residual_r2']].mean()

fig, ax = plt.subplots(figsize=(10, 6))
celltype_overall.plot(kind='barh', stacked=True, ax=ax,
                      color=['#e74c3c', '#3498db', '#95a5a6'])
ax.set_xlabel('Proportion of Variance', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
ax.set_title('Overall Variance Partitioning by Cell Type\n(averaged across all PCs and regions)',
             fontsize=14, fontweight='bold')
ax.legend(['Age', 'Sex', 'Residual/Individual'], loc='lower right')
ax.set_xlim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/variance_partitioning_overall_by_celltype.png', 
            dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/variance_partitioning_overall_by_celltype.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("SUMMARY BY CELL TYPE")
print("="*70)

for ct in cell_types:
    ct_data = results_df[results_df['cell_type'] == ct]
    if len(ct_data) > 0:
        print(f"\n{ct}:")
        print(f"  Age:      {ct_data['age_r2'].mean()*100:.2f}%")
        print(f"  Sex:      {ct_data['sex_r2'].mean()*100:.2f}%")
        print(f"  Residual: {ct_data['residual_r2'].mean()*100:.2f}%")

print("\n" + "="*70)
print("DONE!")
print("="*70)
