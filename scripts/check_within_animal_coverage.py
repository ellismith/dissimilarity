#!/usr/bin/env python
"""
Check which subtypes have enough cells per animal for within-animal analysis
"""

import scanpy as sc
import pandas as pd

# Cell type paths
paths = {
    'Glutamatergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad',
    'GABAergic': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad',
    'Microglia': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_microglia_new.h5ad',
    'Astrocytes': '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad'
}

regions = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

results = []

for cell_type, path in paths.items():
    print(f"\n{'='*70}")
    print(f"Processing {cell_type}...")
    print('='*70)
    
    try:
        adata = sc.read_h5ad(path, backed='r')
        obs_df = adata.obs.copy()
        obs_df['animal_id'] = obs_df['animal_id'].astype(str)
        obs_df['region'] = obs_df['region'].astype(str)
        
        for region in regions:
            # Filter to region and age >= 1
            mask = (obs_df['region'] == region) & (obs_df['age'] >= 1)
            region_data = obs_df[mask]
            
            if len(region_data) == 0:
                continue
            
            # Get subtypes
            subtypes = region_data['ct_louvain'].value_counts()
            
            for subtype, total_cells in subtypes.items():
                # Count cells per animal
                subtype_data = region_data[region_data['ct_louvain'] == subtype]
                animals_per_cell = subtype_data['animal_id'].value_counts()
                
                n_animals_5plus = (animals_per_cell >= 5).sum()
                n_animals_10plus = (animals_per_cell >= 10).sum()
                
                # Only include if at least 5 animals have >= 5 cells
                if n_animals_5plus >= 5:
                    results.append({
                        'cell_type': cell_type,
                        'region': region,
                        'subtype': subtype,
                        'total_cells': total_cells,
                        'n_animals_5plus': n_animals_5plus,
                        'n_animals_10plus': n_animals_10plus,
                        'max_cells_per_animal': animals_per_cell.max()
                    })
                    
                    print(f"{region} - {subtype}: {total_cells} cells, "
                          f"{n_animals_5plus} animals with >=5 cells, "
                          f"{n_animals_10plus} animals with >=10 cells")
        
    except Exception as e:
        print(f"Error processing {cell_type}: {e}")

# Convert to dataframe and save
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['cell_type', 'n_animals_10plus'], ascending=[True, False])

output_file = '/scratch/easmit31/factor_analysis/subtype_variability/within_animal_coverage.csv'
results_df.to_csv(output_file, index=False)

print(f"\n\n{'='*70}")
print("SUMMARY")
print('='*70)
print(f"\nTotal subtypes with >= 5 animals having >= 5 cells: {len(results_df)}")
print(f"\nBy cell type:")
print(results_df.groupby('cell_type').size())

print(f"\nTop 20 by coverage (animals with >= 10 cells):")
print(results_df[['cell_type', 'region', 'subtype', 'n_animals_10plus', 'total_cells']].head(20).to_string())

print(f"\n\nFull results saved to: {output_file}")
