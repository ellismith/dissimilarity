#!/usr/bin/env python

import pandas as pd
import numpy as np
from gprofiler import GProfiler

print("="*70)
print("GENE SET ENRICHMENT FOR ALL 50 PCs AND FACTORS")
print("="*70)

# Load data
pca_loadings = pd.read_csv('pca_all_loadings_with_symbols.csv', index_col=0)
fa_loadings = pd.read_csv('fa_all_loadings_with_symbols.csv', index_col=0)
pca_sources = pd.read_csv('pc_sources_of_variation_50.csv')
fa_sources = pd.read_csv('fa_sources_of_variation_50.csv')

# Initialize gprofiler
gp = GProfiler(return_dataframe=True)

def run_enrichment_for_component(loadings, component_name, top_n=100):
    """Run GO/pathway enrichment on top genes"""
    
    # Get top genes by ABSOLUTE loading
    comp_loadings = loadings.loc[component_name]
    top_genes = comp_loadings.abs().sort_values(ascending=False).head(top_n).index.tolist()
    
    print(f"{component_name}: Testing {len(top_genes)} genes...", end=' ')
    
    try:
        results = gp.profile(
            organism='mmulatta',  # Macaque
            query=top_genes,
            sources=['GO:BP', 'KEGG', 'REAC'],  # Biological Process, KEGG, Reactome
            user_threshold=0.05
        )
        
        if len(results) > 0:
            print(f"Found {len(results)} enriched terms")
            return results
        else:
            print("No significant enrichment")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run enrichment for ALL 50 PCs
print("\n" + "="*70)
print("PCA ENRICHMENT (50 components)")
print("="*70)

pca_enrichments = {}
for i in range(50):
    pc_name = f'PC{i+1}'
    results = run_enrichment_for_component(pca_loadings, pc_name)
    if results is not None:
        pca_enrichments[pc_name] = results

# Save PCA enrichments
all_pca_enrichments = []
for pc_name, enrichment in pca_enrichments.items():
    enrichment['component'] = pc_name
    all_pca_enrichments.append(enrichment)

if all_pca_enrichments:
    pca_enrichment_df = pd.concat(all_pca_enrichments, ignore_index=True)
    pca_enrichment_df.to_csv('pca_go_enrichment_all_50.csv', index=False)
    print(f"\nSaved: pca_go_enrichment_all_50.csv")

# Run enrichment for ALL 50 Factors
print("\n" + "="*70)
print("FACTOR ANALYSIS ENRICHMENT (50 factors)")
print("="*70)

fa_enrichments = {}
for i in range(50):
    factor_name = f'Factor{i+1}'
    results = run_enrichment_for_component(fa_loadings, factor_name)
    if results is not None:
        fa_enrichments[factor_name] = results

# Save FA enrichments
all_fa_enrichments = []
for factor_name, enrichment in fa_enrichments.items():
    enrichment['component'] = factor_name
    all_fa_enrichments.append(enrichment)

if all_fa_enrichments:
    fa_enrichment_df = pd.concat(all_fa_enrichments, ignore_index=True)
    fa_enrichment_df.to_csv('fa_go_enrichment_all_50.csv', index=False)
    print(f"\nSaved: fa_go_enrichment_all_50.csv")

# Create summary with top term per component
print("\n" + "="*70)
print("CREATING SUMMARY")
print("="*70)

pca_summary = []
for i in range(50):
    pc_name = f'PC{i+1}'
    pc_row = pca_sources[pca_sources['PC'] == pc_name].iloc[0]
    
    summary = {
        'Component': pc_name,
        'CellType_eta2': pc_row['celltype_eta2'],
        'Region_eta2': pc_row['region_eta2'],
        'Age_r': pc_row['age_r'],
        'Age_p': pc_row['age_p']
    }
    
    if pc_name in pca_enrichments and len(pca_enrichments[pc_name]) > 0:
        top_term = pca_enrichments[pc_name].iloc[0]
        summary['Top_GO_term'] = top_term['name']
        summary['Top_GO_pval'] = top_term['p_value']
        summary['Top_GO_source'] = top_term['source']
    else:
        summary['Top_GO_term'] = 'None'
        summary['Top_GO_pval'] = 1.0
        summary['Top_GO_source'] = 'None'
    
    pca_summary.append(summary)

pca_summary_df = pd.DataFrame(pca_summary)
pca_summary_df.to_csv('pca_interpretation_summary.csv', index=False)
print("Saved: pca_interpretation_summary.csv")

fa_summary = []
for i in range(50):
    factor_name = f'Factor{i+1}'
    fa_row = fa_sources[fa_sources['Factor'] == factor_name].iloc[0]
    
    summary = {
        'Component': factor_name,
        'CellType_eta2': fa_row['celltype_eta2'],
        'Region_eta2': fa_row['region_eta2'],
        'Age_r': fa_row['age_r'],
        'Age_p': fa_row['age_p']
    }
    
    if factor_name in fa_enrichments and len(fa_enrichments[factor_name]) > 0:
        top_term = fa_enrichments[factor_name].iloc[0]
        summary['Top_GO_term'] = top_term['name']
        summary['Top_GO_pval'] = top_term['p_value']
        summary['Top_GO_source'] = top_term['source']
    else:
        summary['Top_GO_term'] = 'None'
        summary['Top_GO_pval'] = 1.0
        summary['Top_GO_source'] = 'None'
    
    fa_summary.append(summary)

fa_summary_df = pd.DataFrame(fa_summary)
fa_summary_df.to_csv('fa_interpretation_summary.csv', index=False)
print("Saved: fa_interpretation_summary.csv")

# Print readable summary
print("\n" + "="*70)
print("PCA COMPONENT INTERPRETATIONS")
print("="*70)

for idx, row in pca_summary_df.iterrows():
    print(f"\n{row['Component']}:")
    print(f"  Captures: CT={row['CellType_eta2']:.2f}, Reg={row['Region_eta2']:.2f}")
    print(f"  Biology: {row['Top_GO_term']}")

print("\n" + "="*70)
print("FA COMPONENT INTERPRETATIONS")
print("="*70)

for idx, row in fa_summary_df.iterrows():
    print(f"\n{row['Component']}:")
    print(f"  Captures: CT={row['CellType_eta2']:.2f}, Reg={row['Region_eta2']:.2f}")
    print(f"  Biology: {row['Top_GO_term']}")

print("\n" + "="*70)
print("Complete!")
print("="*70)

