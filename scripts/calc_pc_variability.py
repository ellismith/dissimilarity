#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse

def calc_pc_variance(pca_scores):
    """Calculate variance of PC scores across cells"""
    return np.var(pca_scores, axis=0)

def calc_radius_of_gyration(pca_scores):
    """Calculate radius of gyration in PC space"""
    centroid = np.mean(pca_scores, axis=0)
    squared_distances = np.sum((pca_scores - centroid)**2, axis=1)
    return np.sqrt(np.mean(squared_distances))

def main():
    parser = argparse.ArgumentParser(description="Calculate PC variability metrics")
    parser.add_argument("--region_file", type=str, required=True,
                        help="H5AD file for region (e.g., HIP.h5ad)")
    parser.add_argument("--cell_type", type=str, required=True,
                        help="Cell type to analyze")
    parser.add_argument("--louvain_cluster", type=str, default=None,
                        help="Specific Louvain cluster (optional)")
    parser.add_argument("--n_pcs", type=int, default=10,
                        help="Number of PCs to use (default: 10)")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/pc_variability",
                        help="Output directory")
    parser.add_argument("--celltype_col", type=str, default="cell_class_annotation")
    parser.add_argument("--cluster_col", type=str, default="louvain")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load h5ad file
    print(f"Loading {args.region_file}...")
    adata = sc.read_h5ad(args.region_file)
    
    # Filter to cell type
    mask = adata.obs[args.celltype_col] == args.cell_type
    if args.louvain_cluster is not None:
        mask &= adata.obs[args.cluster_col].astype(str) == str(args.louvain_cluster)
        level_name = f"{args.cell_type}_louvain{args.louvain_cluster}"
    else:
        level_name = args.cell_type
    
    # Subset the data
    adata_sub = adata[mask].copy()
    
    if adata_sub.n_obs == 0:
        print(f"No cells found for {level_name}")
        return
    
    print(f"Analyzing {adata_sub.n_obs} cells from {level_name}")
    
    # Run PCA if not already present
    if 'X_pca' not in adata_sub.obsm.keys():
        print("Running PCA...")
        sc.pp.normalize_total(adata_sub, target_sum=1e4)
        sc.pp.log1p(adata_sub)
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000)
        sc.pp.scale(adata_sub, max_value=10)
        sc.tl.pca(adata_sub, n_comps=50)
    
    pca_scores = adata_sub.obsm['X_pca'][:, :args.n_pcs]
    
    results = []
    
    # Calculate per individual
    for animal_id in adata_sub.obs['animal_id'].unique():
        animal_mask = adata_sub.obs['animal_id'] == animal_id
        pca_animal = pca_scores[animal_mask]
        age = adata_sub.obs[adata_sub.obs['animal_id'] == animal_id]['age'].iloc[0]
        
        if len(pca_animal) < 10:  # Skip if too few cells
            print(f"  Skipping {animal_id}: only {len(pca_animal)} cells")
            continue
        
        # Calculate variance per PC
        pc_variances = calc_pc_variance(pca_animal)
        
        # Calculate radius of gyration
        rog = calc_radius_of_gyration(pca_animal)
        
        # Store results
        result = {
            'animal_id': animal_id,
            'age': age,
            'n_cells': len(pca_animal),
            'radius_of_gyration': rog,
            'mean_pc_variance': np.mean(pc_variances),
            'total_pc_variance': np.sum(pc_variances)
        }
        
        # Add individual PC variances
        for i, var in enumerate(pc_variances):
            result[f'PC{i+1}_variance'] = var
        
        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    region_name = os.path.basename(args.region_file).replace('.h5ad', '')
    safe_level_name = level_name.replace(' ', '_').replace('/', '_')
    out_file = os.path.join(args.out_dir, f"pc_variability_{region_name}_{safe_level_name}.csv")
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")
    print(f"Computed metrics for {len(df)} individuals")

if __name__ == "__main__":
    main()
