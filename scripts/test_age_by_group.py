#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined PCA results
pca_df = pd.read_csv('pca_combined_all_celltypes.csv')

print("="*70)
print("TESTING AGE EFFECTS WITHIN EACH CELL TYPE/REGION")
print("="*70)

results = []

# For each cell type
for cell_type in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
    ct_data = pca_df[pca_df['cell_type'] == cell_type]
    
    # For each region within that cell type
    for region in ct_data['region'].unique():
        region_data = ct_data[ct_data['region'] == region]
        
        if len(region_data) < 10:
            continue
        
        print(f"\n{cell_type} - {region} (n={len(region_data)}):")
        
        # Test each PC for age correlation
        best_r = 0
        best_pc = None
        best_p = 1
        
        for i in range(1, 21):
            pc = f'PC{i}'
            
            if pc not in region_data.columns:
                continue
            
            r, p = stats.pearsonr(region_data['age'], region_data[pc])
            
            if abs(r) > abs(best_r):
                best_r = r
                best_pc = pc
                best_p = p
            
            if p < 0.05:
                print(f"  {pc}: r={r:.3f}, p={p:.3e}")
        
        results.append({
            'cell_type': cell_type,
            'region': region,
            'n_samples': len(region_data),
            'best_pc': best_pc,
            'best_r': best_r,
            'best_p': best_p,
            'is_significant': best_p < 0.05
        })

# Create summary
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_r', key=abs, ascending=False)

print("\n" + "="*70)
print("SUMMARY: AGE EFFECTS BY CELL TYPE AND REGION")
print("="*70)

sig_results = results_df[results_df['is_significant']]
print(f"\nSignificant age associations: {len(sig_results)}/{len(results_df)}")

print("\nTop 20 age-associated groups:")
print(sig_results.head(20)[['cell_type', 'region', 'best_pc', 'best_r', 'best_p']].to_string(index=False))

# Which PCs are most commonly age-associated?
print("\n" + "="*70)
print("WHICH PCs CAPTURE AGE MOST OFTEN?")
print("="*70)

pc_counts = sig_results['best_pc'].value_counts()
print(pc_counts)

print("\n→ Now these PC numbers are COMPARABLE across cell types/regions!")

# Create heatmap: PC number vs cell type/region
print("\nCreating heatmap...")

# Create matrix of best PC for each group
pivot_data = results_df.pivot_table(
    values='best_r',
    index='region',
    columns='cell_type',
    aggfunc='first'
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot_data, 
            cmap='RdBu_r',
            center=0,
            vmin=-0.6, vmax=0.6,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Age Correlation (best PC)'},
            linewidths=0.5,
            ax=ax)
ax.set_title('Strongest Age Correlation per Cell Type/Region\n(Using Aligned PCs)', 
            fontsize=13, fontweight='bold')
ax.set_xlabel('Cell Type', fontsize=11)
ax.set_ylabel('Region', fontsize=11)
plt.tight_layout()
plt.savefig('aligned_pca_age_correlations.png', dpi=300, bbox_inches='tight')
print("Saved: aligned_pca_age_correlations.png")

# Which PC for each?
pivot_pc = results_df.pivot_table(
    values='best_pc',
    index='region',
    columns='cell_type',
    aggfunc='first'
)

print("\nWhich PC is best for each group:")
print(pivot_pc.to_string())

# Save results
results_df.to_csv('aligned_pca_age_results.csv', index=False)
print("\nSaved: aligned_pca_age_results.csv")

print("\n" + "="*70)
print("KEY QUESTION: Do same PCs capture aging across groups?")
print("="*70)

# For significant results, how often is it the same PC?
for pc_num in range(1, 21):
    pc_name = f'PC{pc_num}'
    n_groups = (sig_results['best_pc'] == pc_name).sum()
    
    if n_groups >= 2:
        print(f"\n{pc_name}: Age-associated in {n_groups} groups")
        groups = sig_results[sig_results['best_pc'] == pc_name][['cell_type', 'region', 'best_r']]
        print(groups.to_string(index=False))

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

