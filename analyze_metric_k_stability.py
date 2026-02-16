import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scanpy as sc
import os

def analyze_metric_k_stability(louvain, region, h5ad_path, n_cells=10, k_values=[5, 10, 15, 20, 30], min_age=1.0):
    """
    Analyze neighbor overlap across metrics and k values for randomly sampled cells
    """
    
    cell_class = os.path.basename(h5ad_path).replace('Res1_', '').replace('_update.h5ad', '').replace('.h5ad', '').replace('_subset', '')
    
    print(f"\n{'='*70}")
    print(f"METRIC & K-VALUE STABILITY ANALYSIS")
    print(f"Cell type: {cell_class}, Louvain: {louvain}, Region: {region}")
    print(f"Testing {n_cells} random cells, k values: {k_values}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    louvain = str(louvain)
    mask = (adata.obs['louvain'] == louvain) & (adata.obs['region'] == region)
    if min_age is not None:
        mask = mask & (adata.obs['age'] >= min_age)
    
    cell_indices = np.where(mask)[0]
    print(f"Found {len(cell_indices)} cells")
    
    X_subset = adata.X[cell_indices, :]
    
    # Filter genes
    if sp.issparse(X_subset):
        gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
    else:
        gene_expressed = (X_subset > 0).sum(axis=0)
    
    pct_expressing = gene_expressed / len(cell_indices)
    genes_pass = pct_expressing >= 0.05
    X_filtered = X_subset[:, genes_pass]
    
    if sp.issparse(X_filtered):
        X_dense = X_filtered.toarray()
    else:
        X_dense = X_filtered
    
    # Z-scored version
    scaler = StandardScaler()
    X_zscore = scaler.fit_transform(X_dense)
    
    print(f"Filtered to {genes_pass.sum()} genes")
    
    # Get metadata
    metadata = adata.obs.iloc[cell_indices].copy().reset_index(drop=True)
    
    # Randomly sample cells
    np.random.seed(42)
    sampled_cells = np.random.choice(len(cell_indices), n_cells, replace=False)
    
    print(f"\nSampled cells: {sampled_cells}")
    print(f"Animals: {metadata.iloc[sampled_cells]['animal_id'].values}")
    print(f"Ages: {metadata.iloc[sampled_cells]['age'].values}")
    
    # Define all metric combinations
    metric_configs = [
        ('Raw_Euclidean', X_dense, 'euclidean'),
        ('Raw_Correlation', X_dense, 'correlation'),
        ('Raw_Cosine', X_dense, 'cosine'),
        ('Zscore_Euclidean', X_zscore, 'euclidean'),
        ('Zscore_Correlation', X_zscore, 'correlation'),
        ('Zscore_Cosine', X_zscore, 'cosine')
    ]
    
    # Compute all distance matrices once
    print(f"\nComputing distance matrices for all metrics...")
    distance_matrices = {}
    for name, X, metric in metric_configs:
        print(f"  {name}...")
        distance_matrices[name] = pairwise_distances(X, metric=metric)
    
    # For each sampled cell and each k value, get neighbors
    results = []
    
    print(f"\n{'='*70}")
    print("Analyzing neighbors for each cell...")
    print(f"{'='*70}\n")
    
    for cell_idx in sampled_cells:
        print(f"Cell {cell_idx} (Animal: {metadata.iloc[cell_idx]['animal_id']}, Age: {metadata.iloc[cell_idx]['age']:.1f})")
        
        # Store neighbors for all metric-k combinations
        neighbors_dict = {}
        
        for k in k_values:
            for metric_name, dist_matrix in distance_matrices.items():
                # Get neighbors
                d = dist_matrix[cell_idx, :].copy()
                d[cell_idx] = np.inf
                knn_indices = set(np.argsort(d)[:k])
                
                key = f"{metric_name}_k{k}"
                neighbors_dict[key] = knn_indices
        
        # Calculate all pairwise overlaps for this cell
        keys = list(neighbors_dict.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                key1, key2 = keys[i], keys[j]
                
                # Calculate Jaccard similarity
                intersection = len(neighbors_dict[key1] & neighbors_dict[key2])
                union = len(neighbors_dict[key1] | neighbors_dict[key2])
                jaccard = intersection / union if union > 0 else 0
                
                # Parse keys
                metric1, k1 = key1.rsplit('_k', 1)
                metric2, k2 = key2.rsplit('_k', 1)
                
                results.append({
                    'cell_idx': cell_idx,
                    'metric1': metric1,
                    'metric2': metric2,
                    'k1': int(k1),
                    'k2': int(k2),
                    'jaccard': jaccard * 100,
                    'same_metric': metric1 == metric2,
                    'same_k': k1 == k2
                })
    
    results_df = pd.DataFrame(results)
    
    # Aggregate across cells
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    # Case 1: Same metric, different k
    same_metric_diff_k = results_df[(results_df['same_metric'] == True) & (results_df['same_k'] == False)]
    print(f"Same metric, different k values:")
    print(f"  Mean overlap: {same_metric_diff_k['jaccard'].mean():.1f}%")
    print(f"  Range: {same_metric_diff_k['jaccard'].min():.1f}% - {same_metric_diff_k['jaccard'].max():.1f}%")
    
    # Case 2: Different metrics, same k
    diff_metric_same_k = results_df[(results_df['same_metric'] == False) & (results_df['same_k'] == True)]
    print(f"\nDifferent metrics, same k value:")
    print(f"  Mean overlap: {diff_metric_same_k['jaccard'].mean():.1f}%")
    print(f"  Range: {diff_metric_same_k['jaccard'].min():.1f}% - {diff_metric_same_k['jaccard'].max():.1f}%")
    
    # Case 3: Different metrics, different k
    diff_metric_diff_k = results_df[(results_df['same_metric'] == False) & (results_df['same_k'] == False)]
    print(f"\nDifferent metrics, different k values:")
    print(f"  Mean overlap: {diff_metric_diff_k['jaccard'].mean():.1f}%")
    print(f"  Range: {diff_metric_diff_k['jaccard'].min():.1f}% - {diff_metric_diff_k['jaccard'].max():.1f}%")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}\n")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Heatmap - average overlap across all metric-k combinations
    ax = fig.add_subplot(gs[0:2, 0:2])
    
    # Create smaller heatmap - just k=10 and k=20
    k_subset = [10, 20]
    subset_configs = []
    for k in k_subset:
        for metric in ['Raw_Euclidean', 'Raw_Cosine', 'Zscore_Euclidean', 'Zscore_Cosine']:
            subset_configs.append((metric, k))
    
    # Build matrix
    matrix_size = len(subset_configs)
    overlap_matrix = np.zeros((matrix_size, matrix_size))
    
    for i, (m1, k1) in enumerate(subset_configs):
        for j, (m2, k2) in enumerate(subset_configs):
            if i == j:
                overlap_matrix[i, j] = 100
            else:
                subset = results_df[
                    (((results_df['metric1'] == m1) & (results_df['metric2'] == m2)) |
                     ((results_df['metric1'] == m2) & (results_df['metric2'] == m1))) &
                    (((results_df['k1'] == k1) & (results_df['k2'] == k2)) |
                     ((results_df['k1'] == k2) & (results_df['k2'] == k1)))
                ]
                if len(subset) > 0:
                    overlap_matrix[i, j] = subset['jaccard'].mean()
    
    labels = [f"{m.replace('_', ' ')}\nk={k}" for m, k in subset_configs]
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
               vmin=0, vmax=100, ax=ax, square=True,
               xticklabels=labels, yticklabels=labels,
               cbar_kws={'label': 'Neighbor Overlap (%)'})
    ax.set_title('Neighbor Overlap Matrix\n(k=10 and k=20 only)', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    # Plot 2: Effect of k value (same metric) - ALL METRICS
    ax = fig.add_subplot(gs[0, 2])
    
    k_effects = []
    for metric in ['Raw_Euclidean', 'Raw_Correlation', 'Raw_Cosine',
                   'Zscore_Euclidean', 'Zscore_Correlation', 'Zscore_Cosine']:
        for i, k1 in enumerate(k_values[:-1]):
            for k2 in k_values[i+1:]:
                subset = results_df[
                    (results_df['metric1'] == metric) &
                    (results_df['metric2'] == metric) &
                    (((results_df['k1'] == k1) & (results_df['k2'] == k2)) |
                     ((results_df['k1'] == k2) & (results_df['k2'] == k1)))
                ]
                if len(subset) > 0:
                    k_effects.append({
                        'metric': metric.replace('_', ' '),
                        'k_diff': abs(k2 - k1),
                        'overlap': subset['jaccard'].mean()
                    })
    
    k_effects_df = pd.DataFrame(k_effects)
    
    colors = {
        'Raw Euclidean': '#1f77b4',
        'Raw Correlation': '#ff7f0e',
        'Raw Cosine': '#2ca02c',
        'Zscore Euclidean': '#d62728',
        'Zscore Correlation': '#9467bd',
        'Zscore Cosine': '#8c564b'
    }
    
    for metric in k_effects_df['metric'].unique():
        data = k_effects_df[k_effects_df['metric'] == metric]
        grouped = data.groupby('k_diff')['overlap'].mean().reset_index()
        ax.plot(grouped['k_diff'], grouped['overlap'],
               marker='o', label=metric, linewidth=2, markersize=6,
               color=colors.get(metric, 'gray'))
    
    ax.set_xlabel('Difference in k values')
    ax.set_ylabel('Mean Neighbor Overlap (%)')
    ax.set_title('Effect of k Value\n(same metric)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 3: Effect of metric (same k)
    ax = fig.add_subplot(gs[1, 2])
    
    metric_comparisons = {
        'Euc vs Cos\n(Raw)': ('Raw_Euclidean', 'Raw_Cosine'),
        'Euc vs Corr\n(Raw)': ('Raw_Euclidean', 'Raw_Correlation'),
        'Cos vs Corr\n(Raw)': ('Raw_Cosine', 'Raw_Correlation'),
        'Raw vs Zscore\n(Euc)': ('Raw_Euclidean', 'Zscore_Euclidean'),
    }
    
    comparison_results = []
    for name, (m1, m2) in metric_comparisons.items():
        for k in k_values:
            subset = results_df[
                (((results_df['metric1'] == m1) & (results_df['metric2'] == m2)) |
                 ((results_df['metric1'] == m2) & (results_df['metric2'] == m1))) &
                (results_df['k1'] == k) &
                (results_df['k2'] == k)
            ]
            if len(subset) > 0:
                comparison_results.append({
                    'comparison': name,
                    'k': k,
                    'overlap': subset['jaccard'].mean()
                })
    
    comp_df = pd.DataFrame(comparison_results)
    
    for comp in comp_df['comparison'].unique():
        data = comp_df[comp_df['comparison'] == comp]
        ax.plot(data['k'], data['overlap'], marker='o', label=comp, linewidth=2)
    
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Neighbor Overlap (%)')
    ax.set_title('Effect of Metric Choice\n(same k)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 4: Distribution across cells
    ax = fig.add_subplot(gs[2, :])
    
    # For k=10, show distribution of overlaps for key comparisons
    k10_data = []
    labels = []
    
    for name, (m1, m2) in metric_comparisons.items():
        subset = results_df[
            (((results_df['metric1'] == m1) & (results_df['metric2'] == m2)) |
             ((results_df['metric1'] == m2) & (results_df['metric2'] == m1))) &
            (results_df['k1'] == 10) &
            (results_df['k2'] == 10)
        ]
        if len(subset) > 0:
            k10_data.append(subset['jaccard'].values)
            labels.append(name)
    
    # Also add same-metric different-k
    for metric in ['Raw_Euclidean', 'Raw_Cosine']:
        subset = results_df[
            (results_df['metric1'] == metric) &
            (results_df['metric2'] == metric) &
            (((results_df['k1'] == 5) & (results_df['k2'] == 20)) |
             ((results_df['k1'] == 20) & (results_df['k2'] == 5)))
        ]
        if len(subset) > 0:
            k10_data.append(subset['jaccard'].values)
            labels.append(f"{metric.split('_')[1]}\nk5 vs k20")
    
    bp = ax.boxplot(k10_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Neighbor Overlap (%)')
    ax.set_title(f'Distribution Across {n_cells} Sampled Cells', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='red', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Metric & K-Value Stability Analysis\n{cell_class}, Louvain {louvain}, {region}',
                fontsize=14, fontweight='bold')
    
    # Save
    output_file = f'/scratch/easmit31/dissimilarity_analysis/metric_k_stability_{cell_class}_louvain{louvain}_{region}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_file}")
    plt.close()
    
    # Save detailed results
    csv_file = f'/scratch/easmit31/dissimilarity_analysis/metric_k_stability_{cell_class}_louvain{louvain}_{region}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Saved detailed results: {csv_file}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--louvain', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--h5ad', type=str, required=True)
    parser.add_argument('--n-cells', type=int, default=10)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 15, 20, 30])
    
    args = parser.parse_args()
    
    analyze_metric_k_stability(args.louvain, args.region, args.h5ad, args.n_cells, args.k_values)
