#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob

# Find all gene loading files with symbols
gene_files = glob.glob('gene_loadings_*_with_symbols.csv')
print(f"Found {len(gene_files)} gene loading files")

# === Known aging gene signatures from literature ===

# 1. Classic aging/longevity genes
aging_longevity = {
    'IGF1', 'IGF1R', 'FOXO1', 'FOXO3', 'SIRT1', 'SIRT2', 'SIRT3', 'SIRT6',
    'MTOR', 'AMPK', 'TP53', 'CDKN2A', 'TERT', 'KLOTHO', 'GHR'
}

# 2. Senescence markers (SASP)
senescence_markers = {
    'CDKN2A', 'CDKN1A', 'IL6', 'IL1B', 'IL8', 'CXCL1', 'CXCL2', 'CCL2',
    'MMP1', 'MMP3', 'IGFBP7', 'PAI1', 'SERPINE1', 'HMGB1'
}

# 3. Inflammation/immune aging genes
inflammation_genes = {
    'IL6', 'IL1B', 'TNF', 'NFKB1', 'NFKB2', 'STAT3', 'CXCL10', 'CCL2',
    'IFNG', 'IFNB1', 'IRF3', 'IRF7', 'TLR4', 'MYD88', 'NLRP3', 'CCL5'
}

# 4. Mitochondrial/oxidative stress
mitochondrial_stress = {
    'SOD1', 'SOD2', 'CAT', 'GPX1', 'NRF2', 'NFE2L2', 'PINK1', 'PARK2',
    'MFN1', 'MFN2', 'OPA1', 'DRP1', 'DNM1L', 'PPARGC1A', 'TFAM'
}

# 5. Synaptic/neuronal aging
synaptic_aging = {
    'SYP', 'SNAP25', 'STX1A', 'VAMP2', 'SYT1', 'DLG4', 'PSD95', 'HOMER1',
    'SHANK1', 'SHANK2', 'SHANK3', 'NRXN1', 'NLGN1', 'NLGN2', 'NLGN3'
}

# 6. Calcium dysregulation
calcium_aging = {
    'CACNA1A', 'CACNA1C', 'CACNA1D', 'CACNA2D1', 'CACNA2D3', 'KCNMA1',
    'ITPR1', 'ITPR2', 'ITPR3', 'RYR1', 'RYR2', 'ATP2A2', 'CAMK2A'
}

# 7. Autophagy/proteostasis
autophagy_genes = {
    'ATG5', 'ATG7', 'BECN1', 'MAP1LC3A', 'MAP1LC3B', 'SQSTM1', 'p62',
    'LAMP1', 'LAMP2', 'TFEB', 'HSPA1A', 'HSPA8', 'HSPA12A', 'HSP90AA1'
}

# 8. DNA damage/repair
dna_damage = {
    'ATM', 'ATR', 'BRCA1', 'BRCA2', 'RAD51', 'XRCC1', 'PARP1', 'PARP2',
    'TP53BP1', 'TP53BP2', 'H2AFX', 'MDC1', 'ERCC1', 'ERCC2', 'XPA', 'XPC'
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

# 11. Microglial activation markers
microglial_markers = {
    'CD68', 'CD74', 'IBA1', 'AIF1', 'P2RY12', 'TMEM119', 'CX3CR1',
    'TREM2', 'CD11B', 'ITGAM', 'TLR4', 'MHC2', 'GFAP'
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
    'Epigenetic Aging': epigenetic_aging,
    'Microglial Activation': microglial_markers
}

print("="*70)
print("COMPARISON TO PUBLISHED AGING SIGNATURES")
print("="*70)

# Function to extract genes from dataframe
def get_top_genes(file, n=100):
    df = pd.read_csv(file)
    df = df[~df['gene_symbol'].isna()]
    df = df[~df['gene_symbol'].str.startswith('ENSMMUG')]
    return set(df.nlargest(n, 'abs_loading')['gene_symbol'].tolist())

# Check each file against known signatures
results = []

for file in gene_files:
    # Extract info from filename
    parts = file.replace('gene_loadings_', '').replace('_with_symbols.csv', '').split('_')
    
    if len(parts) < 3:
        continue
    
    cell_type = parts[0]
    region = parts[1]
    factor = '_'.join(parts[2:])
    analysis = f"{cell_type}_{region}_{factor}"
    
    all_genes = get_top_genes(file, n=100)
    
    found_any = False
    
    for sig_name, sig_genes in all_signatures.items():
        overlap = all_genes & sig_genes
        
        if overlap:
            found_any = True
            
            results.append({
                'cell_type': cell_type,
                'region': region,
                'factor': factor,
                'analysis': analysis,
                'signature': sig_name,
                'n_overlap': len(overlap),
                'genes_overlap': ', '.join(sorted(overlap))
            })

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('aging_signature_comparison_all_celltypes.csv', index=False)
    print(f"\nSaved: aging_signature_comparison_all_celltypes.csv")
    print(f"Total signature hits: {len(results_df)}")
    
    # === Summary statistics ===
    print("\n" + "="*70)
    print("SUMMARY BY CELL TYPE")
    print("="*70)
    
    for ct in ['Glutamatergic', 'GABAergic', 'Astrocytes', 'Microglia']:
        ct_data = results_df[results_df['cell_type'] == ct]
        if not ct_data.empty:
            print(f"\n{ct}:")
            print(f"  Total signature hits: {len(ct_data)}")
            print(f"  Regions covered: {ct_data['region'].nunique()}")
            
            # Top signatures
            top_sigs = ct_data['signature'].value_counts().head(3)
            print(f"  Top signatures:")
            for sig, count in top_sigs.items():
                print(f"    {sig}: {count} hits")
    
    # === Summary by signature ===
    print("\n" + "="*70)
    print("SUMMARY BY SIGNATURE")
    print("="*70)
    
    sig_summary = results_df.groupby('signature').agg({
        'n_overlap': 'sum',
        'analysis': 'count'
    }).sort_values('n_overlap', ascending=False)
    sig_summary.columns = ['Total_genes', 'N_analyses']
    print(sig_summary.to_string())
    
    # === Key findings ===
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # IGF1/IGF1R
    igf_hits = results_df[results_df['genes_overlap'].str.contains('IGF1', na=False)]
    if not igf_hits.empty:
        print("\n✓ IGF1/IGF1R pathway genes found:")
        for _, row in igf_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Calcium genes
    ca_hits = results_df[results_df['signature'] == 'Calcium Dysregulation']
    if not ca_hits.empty:
        print(f"\n✓ Calcium dysregulation genes found in {len(ca_hits)} analyses:")
        for _, row in ca_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Epigenetic
    epi_hits = results_df[results_df['signature'] == 'Epigenetic Aging']
    if not epi_hits.empty:
        print(f"\n✓ Epigenetic aging markers found in {len(epi_hits)} analyses:")
        for _, row in epi_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Tau/AD
    tau_hits = results_df[results_df['signature'] == "Tau/Alzheimer's"]
    if not tau_hits.empty:
        print(f"\n✓ Tau/AD genes found in {len(tau_hits)} analyses:")
        for _, row in tau_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Inflammation
    inflam_hits = results_df[results_df['signature'] == 'Inflammation']
    if not inflam_hits.empty:
        print(f"\n✓ Inflammatory genes found in {len(inflam_hits)} analyses:")
        for _, row in inflam_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Microglial markers
    micro_hits = results_df[results_df['signature'] == 'Microglial Activation']
    if not micro_hits.empty:
        print(f"\n✓ Microglial activation markers found in {len(micro_hits)} analyses:")
        for _, row in micro_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")
    
    # Synaptic
    syn_hits = results_df[results_df['signature'] == 'Synaptic Aging']
    if not syn_hits.empty:
        print(f"\n✓ Synaptic aging genes found in {len(syn_hits)} analyses:")
        for _, row in syn_hits.iterrows():
            print(f"  {row['cell_type']} - {row['region']}: {row['genes_overlap']}")

else:
    print("\nNo overlaps found with known aging signatures")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

