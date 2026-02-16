import scanpy as sc
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python check_available_combinations.py <cell_type>")
    sys.exit(1)

cell_type = sys.argv[1]

# Map to h5ad filename
h5ad_map = {
    'astrocytes': 'Res1_astrocytes_update.h5ad',
    'GABAergic-neurons': 'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons': 'Res1_glutamatergic-neurons_update.h5ad',
    'medium-spiny-neurons': 'Res1_medium-spiny-neurons_subset.h5ad',
    'opc-olig': 'Res1_opc-olig_subset.h5ad',
    'vascular-cells': 'Res1_vascular-cells_subset.h5ad'
}

if cell_type not in h5ad_map:
    print(f"Unknown cell type: {cell_type}")
    print(f"Available: {list(h5ad_map.keys())}")
    sys.exit(1)

h5ad_path = f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{h5ad_map[cell_type]}'

print(f"Checking {cell_type}...")
print(f"Loading {h5ad_path}...")

adata = sc.read_h5ad(h5ad_path, backed='r')

# Get combinations with age >= 1, >=30 animals, >=100 cells
summary = []
for (louvain, region), group in adata.obs.groupby(['louvain', 'region'], observed=True):
    group_filtered = group[group['age'] >= 1]
    n_cells = len(group_filtered)
    n_animals = group_filtered['animal_id'].nunique()
    
    if n_animals >= 30 and n_cells >= 100:
        summary.append({
            'louvain': louvain,
            'region': region,
            'n_cells': n_cells,
            'n_animals': n_animals
        })

summary_df = pd.DataFrame(summary).sort_values(['louvain', 'region'])

print(f"\n{cell_type} combinations (age ≥ 1, ≥30 animals, ≥100 cells):")
print("="*70)
print(summary_df.to_string(index=False))
print(f"\nTotal combinations: {len(summary_df)}")

# Save
output_file = f'/scratch/easmit31/dissimilarity_analysis/{cell_type}_combinations.csv'
summary_df.to_csv(output_file, index=False)
print(f"✓ Saved to {output_file}")
