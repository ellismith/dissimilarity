#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# Load the GO enrichment results
results_df = pd.read_csv('go_enrichment_all_results.csv')

print(f"Total GO terms found: {len(results_df)}")
print(f"\nAnalyses included: {results_df['analysis'].unique()}")

# Filter to significant terms (adjusted p < 0.05)
sig_results = results_df[results_df['adjusted_pval'] < 0.05].copy()
print(f"\nSignificant terms (p < 0.05): {len(sig_results)}")

# === 1. Find overlapping GO terms across analyses ===
print("\n" + "="*70)
print("OVERLAPPING GO TERMS ACROSS ANALYSES")
print("="*70)

# Count how many analyses each term appears in
term_counts = sig_results.groupby('term').agg({
    'analysis': lambda x: list(x),
    'adjusted_pval': 'min'
}).reset_index()
term_counts['n_analyses'] = term_counts['analysis'].apply(len)
term_counts = term_counts.sort_values('n_analyses', ascending=False)

# Show terms appearing in multiple analyses
shared_terms = term_counts[term_counts['n_analyses'] > 1]
if not shared_terms.empty:
    print("\nTerms appearing in multiple analyses:")
    for idx, row in shared_terms.iterrows():
        print(f"\n{row['term'][:70]}")
        print(f"  Found in: {', '.join(row['analysis'])}")
        print(f"  Best p-value: {row['adjusted_pval']:.2e}")
else:
    print("\nNo terms shared across multiple analyses (very region/cell-type specific)")

# === 2. Create heatmap of top terms ===
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

# Get top 5 terms per analysis (biological process only for clarity)
bp_results = sig_results[sig_results['category'] == 'GO_Biological_Process_2023'].copy()

top_terms_per_analysis = []
for analysis in bp_results['analysis'].unique():
    analysis_df = bp_results[bp_results['analysis'] == analysis].nsmallest(5, 'adjusted_pval')
    top_terms_per_analysis.append(analysis_df)

if top_terms_per_analysis:
    top_bp = pd.concat(top_terms_per_analysis)
    
    # Create a matrix for heatmap
    # Rows = unique terms, Columns = analyses
    all_analyses = ['GABAergic_dlPFC_Factor10', 'GABAergic_ACC_Factor7', 'GABAergic_M1_Factor3',
                   'Astrocytes_CN_Factor7', 'Astrocytes_M1_Factor8', 'Astrocytes_ACC_Factor6']
    
    unique_terms = top_bp['term'].unique()
    
    # Create matrix of -log10(p-values)
    heatmap_data = pd.DataFrame(0, index=unique_terms, columns=all_analyses)
    
    for idx, row in top_bp.iterrows():
        heatmap_data.loc[row['term'], row['analysis']] = -np.log10(row['adjusted_pval'])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(unique_terms)*0.4)))
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                cmap='YlOrRd', 
                cbar_kws={'label': '-log10(adjusted p-value)'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    ax.set_xlabel('Analysis', fontsize=12, fontweight='bold')
    ax.set_ylabel('GO Biological Process Terms', fontsize=12, fontweight='bold')
    ax.set_title('Top Age-Associated GO Terms Across Cell Types and Regions', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Shorten y-axis labels if too long
    yticklabels = [label.get_text()[:60] + '...' if len(label.get_text()) > 60 
                   else label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('go_enrichment_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: go_enrichment_heatmap.png")

# === 3. Bar plot of top terms per analysis ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

analyses_info = [
    ('GABAergic_dlPFC_Factor10', 'GABAergic\ndlPFC\n(r=-0.65)'),
    ('GABAergic_ACC_Factor7', 'GABAergic\nACC\n(r=-0.54)'),
    ('GABAergic_M1_Factor3', 'GABAergic\nM1\n(r=-0.49)'),
    ('Astrocytes_CN_Factor7', 'Astrocytes\nCN\n(r=-0.58)'),
    ('Astrocytes_M1_Factor8', 'Astrocytes\nM1\n(r=0.52)'),
    ('Astrocytes_ACC_Factor6', 'Astrocytes\nACC\n(r=-0.48)')
]

for idx, (analysis, title) in enumerate(analyses_info):
    ax = axes[idx]
    
    # Get top 5 biological process terms
    analysis_bp = bp_results[bp_results['analysis'] == analysis].nsmallest(5, 'adjusted_pval')
    
    if not analysis_bp.empty:
        # Shorten term names
        terms = [t[:40] + '...' if len(t) > 40 else t for t in analysis_bp['term'].tolist()]
        pvals = -np.log10(analysis_bp['adjusted_pval'].values)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(terms))
        bars = ax.barh(y_pos, pvals, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=8)
        ax.set_xlabel('-log10(adj. p-value)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        
        # Add significance line at p=0.05
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, linewidth=1)
        
    else:
        ax.text(0.5, 0.5, 'No significant\nterms', ha='center', va='center', 
               transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('go_enrichment_by_factor.png', dpi=300, bbox_inches='tight')
print("Saved: go_enrichment_by_factor.png")

# === 4. Category breakdown ===
fig, ax = plt.subplots(figsize=(10, 6))

category_counts = sig_results.groupby(['analysis', 'category']).size().unstack(fill_value=0)

# Simplify category names
category_counts.columns = [c.replace('GO_', '').replace('_2023', '').replace('_', ' ') 
                          for c in category_counts.columns]

category_counts.plot(kind='bar', stacked=True, ax=ax, 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xlabel('Analysis', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Significant Terms', fontsize=11, fontweight='bold')
ax.set_title('GO Term Categories by Analysis', fontsize=13, fontweight='bold')
ax.legend(title='GO Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('go_categories_breakdown.png', dpi=300, bbox_inches='tight')
print("Saved: go_categories_breakdown.png")

# === 5. Summary table of pathway themes ===
print("\n" + "="*70)
print("PATHWAY THEME SUMMARY")
print("="*70)

# Manually categorize key themes based on GO terms
themes = {
    'Synaptic/Neuronal': ['synap', 'dendrite', 'axon', 'neuron projection', 'postsynaptic'],
    'Tau/Cytoskeleton': ['tau', 'microtubule', 'cytoskeleton'],
    'Immune/Inflammation': ['interferon', 'immune', 'inflammatory', 'dsRNA'],
    'Ion Channels': ['channel', 'cation', 'potassium', 'calcium'],
    'Cell Signaling': ['GTPase', 'kinase', 'signal transduction', 'receptor'],
    'Metabolism': ['insulin', 'metabolic', 'glycosyl'],
    'Cell Death/Clearance': ['apoptosis', 'phagocytosis', 'endopeptidase']
}

theme_summary = []
for analysis in all_analyses:
    analysis_terms = sig_results[sig_results['analysis'] == analysis]['term'].str.lower()
    
    for theme, keywords in themes.items():
        n_terms = sum(any(kw in term for kw in keywords) for term in analysis_terms)
        if n_terms > 0:
            theme_summary.append({
                'analysis': analysis,
                'theme': theme,
                'n_terms': n_terms
            })

if theme_summary:
    theme_df = pd.DataFrame(theme_summary)
    theme_pivot = theme_df.pivot(index='theme', columns='analysis', values='n_terms').fillna(0).astype(int)
    
    print("\nNumber of terms per biological theme:")
    print(theme_pivot.to_string())
    
    theme_pivot.to_csv('go_theme_summary.csv')
    print("\nSaved: go_theme_summary.csv")

print("\n" + "="*70)
print("Visualization complete!")
print("="*70)

