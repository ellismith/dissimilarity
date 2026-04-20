#!/usr/bin/env python

import pandas as pd
import numpy as np

# Load your gene lists
summary_df = pd.read_csv('age_gene_direction_summary.csv')

# === Known aging gene signatures from literature ===

# 1. Classic aging/longevity genes
aging_longevity = {
    'IGF1', 'IGF1R', 'FOXO1', 'FOXO3', 'SIRT1', 'SIRT2', 'SIRT3', 'SIRT6',
    'MTOR', 'AMPK', 'TP53', 'CDKN2A', 'TERT', 'KLOTHO', 'GHR'
}

# 2. Senescence markers (SASP - Senescence Associated Secretory Phenotype)
senescence_markers = {
    'CDKN2A', 'CDKN1A', 'IL6', 'IL1B', 'IL8', 'CXCL1', 'CXCL2', 'CCL2',
    'MMP1', 'MMP3', 'IGFBP7', 'PAI1', 'SERPINE1', 'HMGB1'
}

# 3. Inflammation/immune aging genes
inflammation_genes = {
    'IL6', 'IL1B', 'TNF', 'NFKB1', 'NFKB2', 'STAT3', 'CXCL10', 'CCL2',
    'IFNG', 'IFNB1', 'IRF3', 'IRF7', 'TLR4', 'MYD88', 'NLRP3'
}

# 4. Mitochondrial/oxidative stress
mitochondrial_stress = {
    'SOD1', 'SOD2', 'CAT', 'GPX1', 'NRF2', 'NFE2L2', 'PINK1', 'PARK2',
    'MFN1', 'MFN2', 'OPA1', 'DRP1', 'DNM1L', 'PPARGC1A', 'TFAM'
}

# 5. Synaptic/neuronal aging (brain-specific)
synaptic_aging = {
    'SYP', 'SNAP25', 'STX1A', 'VAMP2', 'SYT1', 'DLG4', 'PSD95', 'HOMER1',
    'SHANK1', 'SHANK2', 'SHANK3', 'NRXN1', 'NLGN1', 'NLGN2', 'NLGN3'
}

# 6. Calcium dysregulation in aging
calcium_aging = {
    'CACNA1A', 'CACNA1C', 'CACNA1D', 'CACNA2D1', 'CACNA2D3', 'KCNMA1',
    'ITPR1', 'ITPR2', 'ITPR3', 'RYR1', 'RYR2', 'ATP2A2', 'CAMK2A'
}

# 7. Autophagy/proteostasis
autophagy_genes = {
    'ATG5', 'ATG7', 'BECN1', 'MAP1LC3A', 'MAP1LC3B', 'SQSTM1', 'p62',
    'LAMP1', 'LAMP2', 'TFEB', 'CLEAR', 'HSPA1A', 'HSPA8', 'HSP90AA1'
}

# 8. DNA damage/repair
dna_damage = {
    'ATM', 'ATR', 'BRCA1', 'BRCA2', 'RAD51', 'XRCC1', 'PARP1', 'PARP2',
    'TP53BP1', 'H2AFX', 'MDC1', 'ERCC1', 'ERCC2', 'XPA', 'XPC'
}

# 9. Tau/AD-related genes
tau_ad_genes = {
    'MAPT', 'APP', 'PSEN1', 'PSEN2', 'APOE', 'TREM2', 'CD33', 'CLU',
    'GSK3B', 'CDK5', 'MARK1', 'MARK2', 'MARK3', 'MARK4'
}

# 10. Epigenetic aging markers
epigenetic_aging = {
    'TET1', 'TET2', 'TET3', 'DNMT1', 'DNMT3A', 'DNMT3B', 'EZH2',
    'HDAC1', 'HDAC2', 'HDAC3', 'KAT2A', 'KAT2B', 'SIRT1', 'SIRT6'
}

# Combine all signatures
all_signatures = {
    'Aging/Longevity': aging_longevity,
    'Senescence/SASP': senescence_markers,
    'Inflammation': inflammation_genes,
    'Mitochondrial/Oxidative Stress': mitochondrial_stress,
    'Synaptic Aging': synaptic_aging,
    'Calcium Dysregulation': calcium_aging,
    'Autophagy/Proteostasis': autophagy_genes,
    'DNA Damage/Repair': dna_damage,
    'Tau/Alzheimer\'s': tau_ad_genes,
    'Epigenetic Aging': epigenetic_aging
}

print("="*70)
print("COMPARISON TO PUBLISHED AGING SIGNATURES")
print("="*70)

# Function to extract genes from comma-separated string
def extract_genes(gene_string):
    if pd.isna(gene_string):
        return set()
    return set(g.strip() for g in gene_string.split(','))

# Check each of your analyses against known signatures
results = []

for idx, row in summary_df.iterrows():
    analysis = row['analysis']
    genes_up = extract_genes(row['genes_increase'])
    genes_down = extract_genes(row['genes_decrease'])
    all_genes = genes_up | genes_down
    
    print(f"\n{analysis}")
    print("-" * 70)
    print(f"Total genes: {len(all_genes)} ({len(genes_up)} up, {len(genes_down)} down)")
    
    found_any = False
    
    for sig_name, sig_genes in all_signatures.items():
        overlap = all_genes & sig_genes
        
        if overlap:
            found_any = True
            # Check direction
            overlap_up = genes_up & sig_genes
            overlap_down = genes_down & sig_genes
            
            print(f"\n  {sig_name}: {len(overlap)} genes")
            if overlap_up:
                print(f"    ↑ with age: {', '.join(sorted(overlap_up))}")
            if overlap_down:
                print(f"    ↓ with age: {', '.join(sorted(overlap_down))}")
            
            results.append({
                'analysis': analysis,
                'signature': sig_name,
                'n_overlap': len(overlap),
                'genes_overlap': ', '.join(sorted(overlap)),
                'n_up': len(overlap_up),
                'n_down': len(overlap_down),
                'genes_up': ', '.join(sorted(overlap_up)),
                'genes_down': ', '.join(sorted(overlap_down))
            })
    
    if not found_any:
        print("  No matches to known aging signatures")

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('aging_signature_comparison.csv', index=False)
    print("\n" + "="*70)
    print("Saved: aging_signature_comparison.csv")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Which signatures are most represented
    sig_counts = results_df.groupby('signature')['n_overlap'].sum().sort_values(ascending=False)
    print("\nMost represented aging signatures across all analyses:")
    for sig, count in sig_counts.items():
        print(f"  {sig}: {count} gene hits")
    
    # Which analyses have most signature overlap
    analysis_counts = results_df.groupby('analysis')['n_overlap'].sum().sort_values(ascending=False)
    print("\nAnalyses with most signature gene overlap:")
    for analysis, count in analysis_counts.items():
        print(f"  {analysis}: {count} hits")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # IGF1 check
    for idx, row in summary_df.iterrows():
        all_genes = extract_genes(row['genes_increase']) | extract_genes(row['genes_decrease'])
        if 'IGF1' in all_genes:
            direction = 'increases' if 'IGF1' in extract_genes(row['genes_increase']) else 'decreases'
            print(f"✓ IGF1 {direction} with age in {row['analysis']}")
    
    # Calcium genes
    print("\nCalcium-related genes found:")
    for idx, row in results_df[results_df['signature'] == 'Calcium Dysregulation'].iterrows():
        print(f"  {row['analysis']}: {row['genes_overlap']}")
    
    # Epigenetic
    epi_results = results_df[results_df['signature'] == 'Epigenetic Aging']
    if not epi_results.empty:
        print("\nEpigenetic aging markers found:")
        for idx, row in epi_results.iterrows():
            print(f"  {row['analysis']}: {row['genes_overlap']}")
    
    # Tau/AD
    tau_results = results_df[results_df['signature'] == "Tau/Alzheimer's"]
    if not tau_results.empty:
        print("\nTau/AD-related genes found:")
        for idx, row in tau_results.iterrows():
            print(f"  {row['analysis']}: {row['genes_overlap']}")

else:
    print("\nNo overlaps found with known aging signatures")

# === Compare to specific high-impact aging papers ===
print("\n" + "="*70)
print("COMPARISON TO SPECIFIC STUDIES")
print("="*70)

# Genes from López-Otín et al. 2023 "Hallmarks of Aging"
hallmarks_genes = {
    'Genomic instability': {'ATM', 'BRCA1', 'PARP1', 'H2AFX'},
    'Telomere attrition': {'TERT', 'TERF1', 'TERF2'},
    'Epigenetic alterations': {'TET2', 'TET3', 'DNMT3A', 'SIRT1', 'SIRT6'},
    'Loss of proteostasis': {'HSPA1A', 'HSPA8', 'HSP90AA1', 'ATG5', 'SQSTM1'},
    'Mitochondrial dysfunction': {'PINK1', 'PARK2', 'SOD2', 'PPARGC1A'},
    'Cellular senescence': {'CDKN2A', 'CDKN1A', 'IL6'},
    'Stem cell exhaustion': {'NOTCH1', 'WNT', 'BMI1'},
    'Altered communication': {'IL6', 'TNF', 'NFKB1', 'IGF1', 'MTOR'},
    'Disabled macroautophagy': {'ATG5', 'ATG7', 'BECN1', 'TFEB'},
    'Chronic inflammation': {'IL6', 'IL1B', 'TNF', 'NFKB1'}
}

print("\nHallmarks of Aging (López-Otín et al. 2023):")
for hallmark, genes in hallmarks_genes.items():
    for idx, row in summary_df.iterrows():
        all_genes = extract_genes(row['genes_increase']) | extract_genes(row['genes_decrease'])
        overlap = all_genes & genes
        if overlap:
            print(f"  {row['analysis']} - {hallmark}: {', '.join(overlap)}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

