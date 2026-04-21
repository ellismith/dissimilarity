#!/usr/bin/env python

import pandas as pd
import gseapy as gp
import glob

# Find all PCA gene loading files
gene_files = glob.glob('/scratch/easmit31/factor_analysis/csv_files/pca_gene_loadings_*_PC*.csv')
print(f"Found {len(gene_files)} PCA gene loading files")

# GO databases to query
gene_sets = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023'
]

all_results = []

for file in gene_files:
    # Extract info from filename
    # Format: pca_gene_loadings_CellType_Region_PC#.csv
    basename = file.split('/')[-1]
    parts = basename.replace('pca_gene_loadings_', '').replace('.csv', '').split('_')
    
    if len(parts) < 3:
        print(f"Skipping {file} - unexpected format")
        continue
    
    cell_type = parts[0]
    region = parts[1]
    pc = parts[2]
    
    name = f"{cell_type}_{region}_{pc}"
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")
    
    # Load gene list
    df = pd.read_csv(file)
    
    # Get top genes (by absolute loading) - use top 200
    top_genes = df.nlargest(200, 'abs_loading')['gene_symbol'].dropna().tolist()
    
    # Remove any remaining Ensembl IDs that didn't map
    top_genes = [g for g in top_genes if not g.startswith('ENSMMUG')]
    
    print(f"Running enrichment with {len(top_genes)} genes...")
    
    # Run enrichment for each GO category
    for gene_set in gene_sets:
        try:
            enr = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gene_set,
                organism='Human',
                outdir=None,
                cutoff=0.05
            )
            
            if enr.results.empty:
                continue
            
            # Get top 5 most significant terms
            top_terms = enr.results.nsmallest(5, 'Adjusted P-value')
            
            if not top_terms.empty:
                print(f"\n  {gene_set} - Top terms:")
                for idx, row in top_terms.iterrows():
                    print(f"    {row['Term'][:60]:<60} (p={row['Adjusted P-value']:.2e})")
            
            # Store results
            for idx, row in top_terms.iterrows():
                all_results.append({
                    'cell_type': cell_type,
                    'region': region,
                    'pc': pc,
                    'analysis': name,
                    'category': gene_set,
                    'term': row['Term'],
                    'adjusted_pval': row['Adjusted P-value'],
                    'combined_score': row['Combined Score'],
                    'genes': row['Genes']
                })
                
        except Exception as e:
            print(f"  Error with {gene_set}: {e}")

# Save all results
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['cell_type', 'region', 'adjusted_pval'])
    results_df.to_csv('go_enrichment_pca_results.csv', index=False)
    print(f"\n{'='*70}")
    print(f"Saved all results to: go_enrichment_pca_results.csv")
    print(f"Total significant terms found: {len(results_df)}")
    
    # Summary by cell type
    print(f"\n{'='*70}")
    print("SUMMARY BY CELL TYPE")
    print(f"{'='*70}")
    
    for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
        ct_results = results_df[results_df['cell_type'] == ct]
        if not ct_results.empty:
            print(f"\n{ct}: {len(ct_results)} significant GO terms across {ct_results['region'].nunique()} regions")
            
            # Most common terms
            top_terms = ct_results['term'].value_counts().head(5)
            if not top_terms.empty:
                print("  Most common terms:")
                for term, count in top_terms.items():
                    print(f"    {term[:50]}: {count} regions")
    
    # Summary by category
    print(f"\n{'='*70}")
    print("SUMMARY BY GO CATEGORY")
    print(f"{'='*70}")
    
    for category in gene_sets:
        cat_results = results_df[results_df['category'] == category]
        print(f"\n{category.replace('GO_', '').replace('_2023', '')}: {len(cat_results)} terms")

else:
    print("\nNo significant enrichment results found.")

print("\nGO enrichment analysis complete!")

