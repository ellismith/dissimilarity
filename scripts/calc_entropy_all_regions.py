#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse
from glob import glob

def adjusted_entropy(proportions):
    """Compute adjusted entropy as per paper.
    Formula: E_s = [-Σ(p_i * log(p_i)) - log(k)] / log(k)
    Range: [-1, 0] where 0 = uniform (max diversity), -1 = single type (min diversity)
    
    Reference: https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.855076/full
    """
    k = len(proportions)
    if k <= 1:
        return -1.0  # Single category = minimum diversity
    
    # Filter out zeros to avoid log(0)
    p_nonzero = proportions[proportions > 0]
    
    # Calculate Shannon entropy
    H = -(p_nonzero * np.log(p_nonzero)).sum()
    
    # Apply formula: (H - log(k)) / log(k)
    return (H - np.log(k)) / np.log(k)

def calc_entropy_for_individual(obs_i, mode="within_region", celltype_col="cell_class_annotation", cluster_col="subtype"):
    results = []
    age = obs_i['age'].iloc[0]
    
    if mode == "within_region":
        counts = obs_i[celltype_col].value_counts()
        p = counts / counts.sum()
        H_adj = adjusted_entropy(p)
        results.append({
            "animal_id": obs_i['animal_id'].iloc[0],
            "age": age,
            "cell_type": "all",
            "adjusted_entropy": H_adj
        })
    elif mode == "within_cell_type":
        for ct in obs_i[celltype_col].unique():
            sub = obs_i[obs_i[celltype_col] == ct]
            counts = sub[cluster_col].value_counts()
            p = counts / counts.sum()
            H_adj = adjusted_entropy(p)
            results.append({
                "animal_id": obs_i['animal_id'].iloc[0],
                "age": age,
                "cell_type": ct,
                "adjusted_entropy": H_adj
            })
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate adjusted Shannon entropy for all regions and individuals.")
    parser.add_argument("--data_dir", type=str, default="/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update", help="Directory with region h5ad files")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/entropy_csvs", help="Directory to save CSVs")
    parser.add_argument("--mode", type=str, default="within_region", choices=["within_region", "within_cell_type"])
    parser.add_argument("--celltype_col", type=str, default="cell_class_annotation")
    parser.add_argument("--cluster_col", type=str, default="subtype")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    h5_files = glob(os.path.join(args.data_dir, "*.h5ad"))
    
    for f in h5_files:
        region_name = os.path.basename(f).replace(".h5ad", "")
        print(f"Processing region: {region_name}")
        adata = sc.read_h5ad(f, backed="r")
        all_results = []
        
        for individual in adata.obs["animal_id"].unique():
            obs_i = adata.obs[adata.obs["animal_id"] == individual]
            all_results.extend(calc_entropy_for_individual(obs_i, mode=args.mode, celltype_col=args.celltype_col, cluster_col=args.cluster_col))
        
        df = pd.DataFrame(all_results)
        out_file = os.path.join(args.out_dir, f"entropy_{region_name}.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved CSV: {out_file}")

if __name__ == "__main__":
    main()
