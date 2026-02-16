import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os
import argparse

def compute_all_distance_metrics(louvain, region, h5ad_path, min_age=1.0):
    """
    Compare all distance metric combinations:
    - Raw: Euclidean, Correlation, Cosine
    - Z-scored: Euclidean, Correlation, Cosine
    """
    
    # Extract cell type
    cell_class = os.path.basename(h5ad_path).replace('Res1_', '').replace('_update.h5ad', '').replace('.h5ad', '').replace('_subset', '').replace('_new', '')
    
    print(f"\n{'='*70}")
    print(f"DISTANCE METRIC COMPARISON")
    print(f"Cell type: {cell_class}, Louvain: {louvain}, Region: {region}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    # Select cells
    louvain = str(louvain)
    mask = (adata.obs['louvain'] == louvain) & (adata.obs['region'] == region)
    if min_age is not None:
        mask = mask & (adata.obs['age'] >= min_age)
    
    cell_indices = np.where(mask)[0]
    
    if len(cell_indices) == 0:
        print("ERROR: No cells found!")
        return None
    
    print(f"Found {len(cell_indices)} cells")
    
    # Load expression
    X_subset = adata.X[cell_indices, :]
    
    # Filter genes
    if sp.issparse(X_subset):
        gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
    else:
        gene_expressed = (X_subset > 0).sum(axis=0)
    
    pct_expressing = gene_expressed / len(cell_indices)
    genes_pass = pct_expressing >= 0.05
    
    X_filtered = X_subset[:, genes_pass]
    
    print(f"Filtered to {genes_pass.sum()} genes")
    
    # Convert to dense
    if sp.issparse(X_filtered):
        X_dense = X_filtered.toarray()
    else:
        X_dense = X_filtered
    
    # Create z-scored version
    scaler = StandardScaler()
    X_zscore = scaler.fit_transform(X_dense)
    
    print(f"Matrix shape: {X_dense.shape}")
    print(f"Memory: {X_dense.nbytes / 1e9:.2f} GB\n")
    
    # Compute all distance matrices
    results = []
    
    for data_type, X in [('Raw', X_dense), ('Z-scored', X_zscore)]:
        for metric in ['euclidean', 'correlation', 'cosine']:
            print(f"Computing {data_type} + {metric}...")
            
            dist = pairwise_distances(X, metric=metric)
            
            # Calculate structure metrics
            nn_dists = []
            for i in range(len(dist)):
                d = dist[i].copy()
                d[i] = np.inf
                nn_dists.append(d.min())
            
            mean_nn = np.mean(nn_dists)
            random_dist = dist[np.triu_indices_from(dist, k=1)].mean()
            ratio = random_dist / mean_nn
            
            # Determine quality
            if ratio > 2.0:
                quality = "Excellent"
            elif ratio > 1.5:
                quality = "Good"
            elif ratio > 1.3:
                quality = "Weak"
            else:
                quality = "Poor"
            
            results.append({
                'data_type': data_type,
                'metric': metric,
                'mean_nn_dist': mean_nn,
                'mean_random_dist': random_dist,
                'ratio': ratio,
                'quality': quality
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Pivot table for easy comparison
    pivot = results_df.pivot(index='metric', columns='data_type', values='ratio')
    
    print("Structure Ratio (Random/NN) - Higher is Better:")
    print(pivot.to_string())
    
    # Find best overall
    best_idx = results_df['ratio'].idxmax()
    best = results_df.iloc[best_idx]
    
    print(f"\n{'='*70}")
    print("WINNER:")
    print(f"{'='*70}")
    print(f"{best['data_type']} + {best['metric']}: {best['ratio']:.2f}x")
    print(f"Quality: {best['quality']}")
    
    # Print all results with quality
    print(f"\n{'='*70}")
    print("DETAILED RESULTS:")
    print(f"{'='*70}\n")
    
    for _, row in results_df.sort_values('ratio', ascending=False).iterrows():
        print(f"{row['data_type']:10s} + {row['metric']:12s}: "
              f"{row['ratio']:.2f}x  ({row['quality']:10s})  "
              f"[NN={row['mean_nn_dist']:.2f}, Random={row['mean_random_dist']:.2f}]")
    
    # Key insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}\n")
    
    # Compare raw vs zscore for euclidean
    raw_euc = results_df[(results_df['data_type'] == 'Raw') & (results_df['metric'] == 'euclidean')].iloc[0]
    zscore_euc = results_df[(results_df['data_type'] == 'Z-scored') & (results_df['metric'] == 'euclidean')].iloc[0]
    
    if raw_euc['ratio'] > zscore_euc['ratio']:
        print(f"• Raw expression outperforms z-scored ({raw_euc['ratio']:.2f} vs {zscore_euc['ratio']:.2f})")
        print(f"  → Magnitude information is valuable")
    else:
        print(f"• Z-scoring improves structure ({zscore_euc['ratio']:.2f} vs {raw_euc['ratio']:.2f})")
        print(f"  → Batch effects present, normalization helps")
    
    # Compare metrics on raw
    raw_results = results_df[results_df['data_type'] == 'Raw'].sort_values('ratio', ascending=False)
    best_raw = raw_results.iloc[0]
    
    print(f"\n• Best metric on raw data: {best_raw['metric']} ({best_raw['ratio']:.2f}x)")
    
    if best_raw['metric'] == 'euclidean':
        print(f"  → Magnitude matters for biology")
    else:
        print(f"  → Pattern similarity dominates (magnitude less important)")
    
    # Return results
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare all distance metrics')
    parser.add_argument('--louvain', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--h5ad', type=str, required=True)
    parser.add_argument('--min-age', type=float, default=1.0)
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (optional)')
    
    args = parser.parse_args()
    
    results_df = compute_all_distance_metrics(
        louvain=args.louvain,
        region=args.region,
        h5ad_path=args.h5ad,
        min_age=args.min_age
    )
    
    if results_df is not None and args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\n✓ Saved results to {args.output}")
