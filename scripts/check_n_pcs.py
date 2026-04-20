#!/usr/bin/env python
import pandas as pd
import glob

pca_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_analysis_*.csv')

print(f"Checking {len(pca_files)} PCA files...\n")

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
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        n_pcs = len(pc_cols)
        n_animals = len(df)
        
        results.append({
            'cell_type': cell_type,
            'region': region,
            'n_animals': n_animals,
            'n_pcs': n_pcs
        })
        
        print(f"{cell_type:20s} {region:6s}: {n_animals:3d} animals, {n_pcs:2d} PCs")
    
    except Exception as e:
        print(f"ERROR: {basename} - {e}")

# Summary
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nNumber of PCs distribution:")
print(results_df['n_pcs'].value_counts().sort_index())
print(f"\nMin PCs: {results_df['n_pcs'].min()}")
print(f"Max PCs: {results_df['n_pcs'].max()}")
print(f"Mean PCs: {results_df['n_pcs'].mean():.1f}")
