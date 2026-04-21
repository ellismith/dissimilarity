#!/usr/bin/env python

import pandas as pd
import scanpy as sc
import glob

# Load one file to get gene name mapping
print("Loading data to get gene name mapping...")
adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

# Create mapping from Ensembl ID to gene symbol
gene_mapping = adata.var['external_gene_name'].to_dict()
print(f"Created mapping for {len(gene_mapping)} genes")

# Find all gene loading files
gene_files = glob.glob('gene_loadings_*.csv')
print(f"\nFound {len(gene_files)} gene loading files to process")

all_summaries = []

for file in gene_files:
    # Skip files that already have symbols
    if 'with_symbols' in file:
        continue
    
    try:
        df = pd.read_csv(file)
        
        # Check if gene_symbol column already exists
        if 'gene_symbol' in df.columns:
            print(f"Skipping {file} - already has symbols")
            continue
            
        df['gene_symbol'] = df['gene'].map(gene_mapping)
        
        # Move gene_symbol to second column
        cols = ['gene', 'gene_symbol', 'loading', 'abs_loading']
        df = df[cols]
        
        # Save with symbols
        output_file = file.replace('.csv', '_with_symbols.csv')
        df.to_csv(output_file, index=False)
        
        # Extract info from filename
        # Format: gene_loadings_CellType_Region_Factor#.csv
        parts = file.replace('gene_loadings_', '').replace('.csv', '').split('_')
        
        if len(parts) >= 3:
            cell_type = parts[0]
            region = parts[1]
            factor = parts[2] if len(parts) == 3 else '_'.join(parts[2:])
            
            print(f"\n{'='*70}")
            print(f"{cell_type} - {region} - {factor}")
            print(f"{'='*70}")
            print("Top 20 genes:")
            print(df[['gene_symbol', 'loading']].head(20).to_string(index=False))
            
            # Store summary
            all_summaries.append({
                'cell_type': cell_type,
                'region': region,
                'factor': factor,
                'top_10_genes': ', '.join(df['gene_symbol'].head(10).fillna('NA').tolist())
            })
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Create readable summary
if all_summaries:
    print("\n" + "="*70)
    print("SUMMARY: TOP 10 GENES PER AGE-ASSOCIATED FACTOR")
    print("="*70)
    summary_df = pd.DataFrame(all_summaries)
    for idx, row in summary_df.iterrows():
        print(f"\n{row['cell_type']} - {row['region']} - {row['factor']}:")
        print(f"  {row['top_10_genes']}")
    
    summary_df.to_csv('age_gene_programs_summary_all_celltypes.csv', index=False)
    print("\n" + "="*70)
    print("Saved readable summary to: age_gene_programs_summary_all_celltypes.csv")
    print(f"Processed {len(gene_files)} gene loading files")
    print("Individual gene lists with symbols saved as: *_with_symbols.csv")
    print("="*70)

