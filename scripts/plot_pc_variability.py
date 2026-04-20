#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob

def plot_scatter(df, metric, title, out_file):
    """Plot scatter of age vs metric with trend line"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(df['age'], df[metric], alpha=0.6, s=60, color='steelblue')
    
    # Add trend line
    z = np.polyfit(df['age'], df[metric], 1)
    p = np.poly1d(z)
    ax.plot(df['age'], p(df['age']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(df['age'], df[metric])[0, 1]
    
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f"{title}\nPearson r = {corr:.3f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def plot_pc_variances(df, title, out_file, n_pcs=10):
    """Plot individual PC variances vs age"""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    
    for i in range(n_pcs):
        pc_col = f'PC{i+1}_variance'
        if pc_col not in df.columns:
            continue
            
        ax = axes[i]
        
        # Scatter
        ax.scatter(df['age'], df[pc_col], alpha=0.6, s=40, color='steelblue')
        
        # Trend line
        z = np.polyfit(df['age'], df[pc_col], 1)
        p = np.poly1d(z)
        ax.plot(df['age'], p(df['age']), "r--", alpha=0.8, linewidth=1.5)
        
        # Correlation
        corr = np.corrcoef(df['age'], df[pc_col])[0, 1]
        
        ax.set_xlabel("Age", fontsize=9)
        ax.set_ylabel(f"PC{i+1} Variance", fontsize=9)
        ax.set_title(f"PC{i+1} (r={corr:.3f})", fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot PC variability metrics")
    parser.add_argument("--csv_dir", type=str, default="/scratch/easmit31/variability/pc_variability",
                        help="Directory with PC variability CSV files")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/pc_variability_figs",
                        help="Output directory for figures")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob(os.path.join(args.csv_dir, "pc_variability_*.csv"))
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).replace('pc_variability_', '').replace('.csv', '')
        print(f"\nProcessing {basename}...")
        
        df = pd.read_csv(csv_file)
        
        # Plot radius of gyration
        out_rog = os.path.join(args.out_dir, f"scatter_rog_{basename}.png")
        plot_scatter(df, 'radius_of_gyration', 
                    f"Radius of Gyration vs Age - {basename.replace('_', ' ')}", 
                    out_rog)
        
        # Plot mean PC variance
        out_mean = os.path.join(args.out_dir, f"scatter_mean_pc_var_{basename}.png")
        plot_scatter(df, 'mean_pc_variance', 
                    f"Mean PC Variance vs Age - {basename.replace('_', ' ')}", 
                    out_mean)
        
        # Plot total PC variance
        out_total = os.path.join(args.out_dir, f"scatter_total_pc_var_{basename}.png")
        plot_scatter(df, 'total_pc_variance', 
                    f"Total PC Variance vs Age - {basename.replace('_', ' ')}", 
                    out_total)
        
        # Plot individual PC variances
        out_pcs = os.path.join(args.out_dir, f"scatter_individual_pcs_{basename}.png")
        plot_pc_variances(df, f"Individual PC Variances vs Age - {basename.replace('_', ' ')}", 
                         out_pcs, n_pcs=10)

if __name__ == "__main__":
    main()
