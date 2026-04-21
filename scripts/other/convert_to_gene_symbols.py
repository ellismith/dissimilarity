#!/usr/bin/env python

import pandas as pd
import scanpy as sc

# Load one of the h5ad files to get the gene name mapping
print("Loading data to get gene name mapping...")
adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

# Create mapping from Ensembl ID to gene symbol
gene_mapping = adata.var['external_gene_name'].to_dict()
print(f"Created mapping for {len(gene_mapping)} genes")

# Files to convert
gene_files = [
    'gene_loadings_GABAergic_dlPFC_Factor10.csv',
    'gene_loadings_GABAergic_ACC_Factor7.csv',
    'gene_loadings_GABAergic_M1_Factor3.csv',
    'gene_loadings_Astrocytes_CN_Factor7.csv',
    'gene_loadings_Astrocytes_M1_Factor8.csv',
    'gene_loadings_Astrocytes_ACC_Factor6.csv'
]

all_summaries = []

for file in gene_files:
    try:
        df = pd.read_csv(file)
        df['gene_symbol'] = df['gene'].map(gene_mapping)
        
        # Move gene_symbol to second column
        cols = ['gene', 'gene_symbol', 'loading', 'abs_loading']
        df = df[cols]
        
        # Save with symbols
        output_file = file.replace('.csv', '_with_symbols.csv')
        df.to_csv(output_file, index=False)
        
        # Extract region and cell type from filename
        parts = file.replace('gene_loadings_', '').replace('.csv', '').split('_')
        cell_type = parts[0]
        region = parts[1]
        factor = parts[2]
        
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
print("\n" + "="*70)
print("SUMMARY: TOP 10 GENES PER AGE-ASSOCIATED FACTOR")
print("="*70)
summary_df = pd.DataFrame(all_summaries)
for idx, row in summary_df.iterrows():
    print(f"\n{row['cell_type']} - {row['region']} - {row['factor']}:")
    print(f"  {row['top_10_genes']}")

summary_df.to_csv('age_gene_programs_summary_readable.csv', index=False)
print("\n" + "="*70)
print("Saved readable summary to: age_gene_programs_summary_readable.csv")
print("Individual gene lists with symbols saved as: *_with_symbols.csv")
print("="*70)

