import pandas as pd
import scanpy as sc

# Load the passing combinations summary
passing_df = pd.read_csv('/scratch/easmit31/dissimilarity_analysis/louvain_region_passing.csv')

# Load GABAergic h5ad to check actual louvain values
h5ad_path = '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_GABAergic-neurons_subset.h5ad'
adata = sc.read_h5ad(h5ad_path, backed='r')

# Get metadata summary for GABAergic neurons
gaba_summary = []
for (louvain, region), group in adata.obs.groupby(['louvain', 'region'], observed=True):
    # Apply age filter >= 1
    group_filtered = group[group['age'] >= 1]
    n_cells = len(group_filtered)
    n_animals = group_filtered['animal_id'].nunique()
    
    if n_animals >= 30 and n_cells >= 100:  # At least 30 animals and 100 cells
        gaba_summary.append({
            'louvain': louvain,
            'region': region,
            'n_cells': n_cells,
            'n_animals': n_animals
        })

gaba_df = pd.DataFrame(gaba_summary)
gaba_df = gaba_df.sort_values(['louvain', 'region'])

print("GABAergic neuron combinations (age ≥ 1, ≥30 animals, ≥100 cells):")
print("="*70)
print(gaba_df.to_string(index=False))
print(f"\nTotal combinations: {len(gaba_df)}")

# Save for reference
gaba_df.to_csv('/scratch/easmit31/dissimilarity_analysis/gabaergic_combinations.csv', index=False)
print(f"\n✓ Saved to gabaergic_combinations.csv")

# Print in format for bash array
print("\n" + "="*70)
print("For batch script COMBINATIONS array:")
print("="*70)
for _, row in gaba_df.iterrows():
    print(f'    "{row["louvain"]}:{row["region"]}"')
