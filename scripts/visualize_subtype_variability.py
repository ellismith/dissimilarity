#!/usr/bin/env python
"""
Visualize subtype variability results - focus on subtype-level patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize subtype variability patterns')
    parser.add_argument('--cell_types', type=str, nargs='+', 
                        default=['Glutamatergic', 'GABAergic'],
                        help='Cell types to visualize (e.g., Glutamatergic GABAergic Microglia)')
    parser.add_argument('--results_dir', type=str, 
                        default='/scratch/easmit31/factor_analysis/subtype_variability',
                        help='Directory containing results CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/easmit31/factor_analysis/subtype_variability/figures',
                        help='Output directory for figures')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print("VISUALIZING SUBTYPE VARIABILITY PATTERNS")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print("="*70)
    
    # Load all results files
    result_files = glob.glob(f"{args.results_dir}/subtype_variability_*.csv")
    
    print(f"\nFound {len(result_files)} result files")
    
    # Load and combine
    all_results = []
    for f in result_files:
        df = pd.read_csv(f)
        all_results.append(df)
    
    combined = pd.concat(all_results, ignore_index=True)
    print(f"Total results: {len(combined)} subtype-PC-region combinations")
    
    # Filter to cell types of interest
    plot_data = combined[combined['cell_type'].isin(args.cell_types)].copy()
    print(f"\nFiltered to specified cell types: {len(plot_data)} combinations")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For each subtype, get the strongest correlation across all PCs and regions
    print("\nCalculating strongest correlation per subtype...")
    subtype_max = plot_data.loc[plot_data.groupby(['cell_type', 'subtype'])['abs_dev_corr'].idxmax()]
    
    # Figure 1: Heatmap of correlation strengths by subtype and region
    print("\n" + "="*70)
    print("Creating Figure 1: Subtype correlation heatmaps")
    print("="*70)
    
    for cell_type in args.cell_types:
        data = plot_data[plot_data['cell_type'] == cell_type]
        
        if len(data) == 0:
            print(f"No data for {cell_type}")
            continue
        
        # Average correlation across all PCs for each subtype-region
        pivot_data = data.groupby(['subtype', 'region'])['abs_dev_corr'].mean().reset_index()
        pivot = pivot_data.pivot(index='subtype', columns='region', values='abs_dev_corr')
        
        # Sort subtypes by overall correlation strength
        pivot['mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.3)))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Mean Correlation (r)'}, ax=ax)
        ax.set_title(f'{cell_type} Subtypes\nMean Correlation Across All PCs', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Subtype', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{cell_type}_subtype_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {args.output_dir}/{cell_type}_subtype_heatmap.png")
        plt.close()
    
    # Figure 2: Strongest correlation per subtype
    print("\n" + "="*70)
    print("Creating Figure 2: Top subtypes by correlation strength")
    print("="*70)
    
    for cell_type in args.cell_types:
        data = subtype_max[subtype_max['cell_type'] == cell_type].copy()
        
        if len(data) == 0:
            print(f"No data for {cell_type}")
            continue
        
        # Sort by absolute correlation
        data = data.sort_values('abs_dev_corr', key=abs, ascending=True)
        
        # Color by direction
        colors = ['red' if x > 0 else 'blue' for x in data['abs_dev_corr']]
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(data) * 0.3)))
        y_pos = range(len(data))
        
        ax.barh(y_pos, data['abs_dev_corr'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['subtype']} ({row['region']}-{row['PC']})" 
                             for _, row in data.iterrows()], fontsize=9)
        ax.set_xlabel('Correlation (r)', fontsize=12)
        ax.set_title(f'{cell_type} Subtypes\nStrongest Correlation (any PC, any region)', 
                     fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Add significance markers
        for i, (_, row) in enumerate(data.iterrows()):
            if row['abs_dev_fdr'] < 0.05:
                ax.text(row['abs_dev_corr'], i, ' ***', 
                       ha='left' if row['abs_dev_corr'] > 0 else 'right',
                       va='center', fontsize=10, fontweight='bold')
        
        ax.legend(handles=[
            plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, label='Increasing'),
            plt.Rectangle((0,0),1,1, fc='blue', alpha=0.7, label='Decreasing')
        ], loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{cell_type}_strongest_per_subtype.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {args.output_dir}/{cell_type}_strongest_per_subtype.png")
        plt.close()
    
    # Save detailed summary
    print("\n" + "="*70)
    print("Creating summary tables")
    print("="*70)
    
    # Summary by subtype
    subtype_summary = plot_data.groupby(['cell_type', 'subtype']).agg({
        'abs_dev_corr': ['mean', 'std', 'min', 'max'],
        'abs_dev_pval': 'min',
        'n_animals': 'mean'
    }).round(3)
    subtype_summary.columns = ['mean_r', 'std_r', 'min_r', 'max_r', 'best_pval', 'mean_n']
    subtype_summary = subtype_summary.reset_index().sort_values('mean_r', key=abs, ascending=False)
    subtype_summary.to_csv(f"{args.output_dir}/subtype_summary.csv", index=False)
    print(f"Saved: {args.output_dir}/subtype_summary.csv")
    
    print("\n" + "="*70)
    print("DONE!")
    print(f"Figures saved to: {args.output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
