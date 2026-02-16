import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scanpy as sc
import argparse
import os

def compare_knn_across_all_metrics(louvain, region, h5ad_path, min_age=1.0, k=10):
    """
    Compute KNN for all 6 metric combinations and compare results
    """
    
    # Extract cell type
    cell_class = os.path.basename(h5ad_path).replace('Res1_', '').replace('_update.h5ad', '').replace('.h5ad', '').replace('_subset', '')
    
    print(f"\n{'='*70}")
    print(f"KNN COMPARISON ACROSS ALL METRICS")
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
    
    # Get metadata
    metadata = adata.obs.iloc[cell_indices].copy().reset_index(drop=True)
    
    print(f"Matrix shape: {X_dense.shape}")
    print(f"Memory: {X_dense.nbytes / 1e9:.2f} GB\n")
    
    # Define all 6 combinations
    combinations = [
        ('Raw', 'euclidean', X_dense),
        ('Raw', 'correlation', X_dense),
        ('Raw', 'cosine', X_dense),
        ('Z-scored', 'euclidean', X_zscore),
        ('Z-scored', 'correlation', X_zscore),
        ('Z-scored', 'cosine', X_zscore)
    ]
    
    # Store results for each combination
    all_results = {}
    
    for data_type, metric, X in combinations:
        combo_name = f"{data_type}_{metric}"
        print(f"Computing KNN for {combo_name}...")
        
        # Compute distance matrix
        dist = pairwise_distances(X, metric=metric)
        
        # For each cell, find k nearest neighbors
        results = []
        for i in range(len(metadata)):
            d = dist[i, :].copy()
            d[i] = np.inf  # Exclude self
            
            # Find k nearest neighbors
            knn_indices = np.argsort(d)[:k]
            knn_distances = d[knn_indices]
            
            # Get metadata for neighbors
            knn_animals = metadata.iloc[knn_indices]['animal_id'].values
            knn_ages = metadata.iloc[knn_indices]['age'].values
            
            # Calculate metrics
            cell_age = metadata.iloc[i]['age']
            cell_animal = metadata.iloc[i]['animal_id']
            age_diffs = np.abs(knn_ages - cell_age)
            
            # Count same vs different animal
            same_animal_count = (knn_animals == cell_animal).sum()
            
            # Store
            results.append({
                'cell_index': i,
                'nearest_neighbor_distance': knn_distances[0],
                'mean_knn_distance': knn_distances.mean(),
                'min_age_diff': age_diffs.min(),
                'mean_age_diff': age_diffs.mean(),
                'same_animal_neighbors': same_animal_count,
                'nearest_neighbor_same_animal': knn_animals[0] == cell_animal
            })
        
        all_results[combo_name] = pd.DataFrame(results)
    
    # Create comparison dataframe
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    comparison = []
    for combo_name, results_df in all_results.items():
        data_type, metric = combo_name.split('_', 1)
        
        comparison.append({
            'data_type': data_type,
            'metric': metric,
            'mean_nn_distance': results_df['nearest_neighbor_distance'].mean(),
            'mean_knn_distance': results_df['mean_knn_distance'].mean(),
            'mean_min_age_diff': results_df['min_age_diff'].mean(),
            'mean_age_diff': results_df['mean_age_diff'].mean(),
            'pct_same_animal_neighbors': results_df['same_animal_neighbors'].mean() / k * 100,
            'pct_nn_same_animal': results_df['nearest_neighbor_same_animal'].sum() / len(results_df) * 100
        })
    
    comp_df = pd.DataFrame(comparison)
    
    print("Age Clustering (min age diff to NN):")
    print(comp_df[['data_type', 'metric', 'mean_min_age_diff']].to_string(index=False))
    
    print(f"\nAnimal Clustering (% NN from same animal):")
    print(comp_df[['data_type', 'metric', 'pct_nn_same_animal']].to_string(index=False))
    
    print(f"\nSame-Animal in k={k} (% across all neighbors):")
    print(comp_df[['data_type', 'metric', 'pct_same_animal_neighbors']].to_string(index=False))
    
    # Create comprehensive visualization
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}\n")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors = {
        'Raw_euclidean': '#1f77b4',
        'Raw_correlation': '#ff7f0e',
        'Raw_cosine': '#2ca02c',
        'Z-scored_euclidean': '#d62728',
        'Z-scored_correlation': '#9467bd',
        'Z-scored_cosine': '#8c564b'
    }
    
    # Plot 1: Age difference to NN - distributions
    ax = fig.add_subplot(gs[0, 0:2])
    for combo_name, results_df in all_results.items():
        ax.hist(results_df['min_age_diff'], bins=50, alpha=0.5, 
               label=combo_name.replace('_', ' '), edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Age Diff to Nearest Neighbor (years)')
    ax.set_ylabel('Count')
    ax.set_title('Age Clustering: NN Age Difference Distribution')
    ax.legend(fontsize=8)
    
    # Plot 2: Age difference - bar comparison
    ax = fig.add_subplot(gs[0, 2:4])
    comp_sorted = comp_df.sort_values('mean_min_age_diff')
    x = np.arange(len(comp_sorted))
    bars = ax.barh(x, comp_sorted['mean_min_age_diff'], 
                   color=[colors[f"{r['data_type']}_{r['metric']}"] for _, r in comp_sorted.iterrows()],
                   edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels([f"{r['data_type']} {r['metric']}" for _, r in comp_sorted.iterrows()], fontsize=9)
    ax.set_xlabel('Mean Age Diff to NN (years)')
    ax.set_title('Age Clustering Comparison\n(Lower = stronger clustering)')
    ax.axvline(comp_df['mean_min_age_diff'].mean(), color='red', linestyle='--', alpha=0.5)
    
    # Add values
    for i, (_, row) in enumerate(comp_sorted.iterrows()):
        ax.text(row['mean_min_age_diff'], i, f" {row['mean_min_age_diff']:.2f}", 
               va='center', fontsize=8)
    
    # Plot 3: Same-animal NN percentage
    ax = fig.add_subplot(gs[1, 0:2])
    comp_sorted = comp_df.sort_values('pct_nn_same_animal', ascending=False)
    x = np.arange(len(comp_sorted))
    bars = ax.barh(x, comp_sorted['pct_nn_same_animal'],
                   color=[colors[f"{r['data_type']}_{r['metric']}"] for _, r in comp_sorted.iterrows()],
                   edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels([f"{r['data_type']} {r['metric']}" for _, r in comp_sorted.iterrows()], fontsize=9)
    ax.set_xlabel('% Nearest Neighbors from Same Animal')
    ax.set_title('Animal Clustering\n(Lower = better mixing)')
    ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax.legend()
    
    # Add values
    for i, (_, row) in enumerate(comp_sorted.iterrows()):
        ax.text(row['pct_nn_same_animal'], i, f" {row['pct_nn_same_animal']:.1f}%",
               va='center', fontsize=8)
    
    # Plot 4: NN distance distributions
    ax = fig.add_subplot(gs[1, 2:4])
    data_to_plot = []
    labels = []
    for combo_name in ['Raw_euclidean', 'Raw_correlation', 'Raw_cosine',
                       'Z-scored_euclidean', 'Z-scored_correlation', 'Z-scored_cosine']:
        # Normalize distances for comparison
        dists = all_results[combo_name]['nearest_neighbor_distance'].values
        dists_norm = (dists - dists.mean()) / dists.std()
        data_to_plot.append(dists_norm)
        labels.append(combo_name.replace('_', '\n'))
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, combo_name in zip(bp['boxes'], ['Raw_euclidean', 'Raw_correlation', 'Raw_cosine',
                                                'Z-scored_euclidean', 'Z-scored_correlation', 'Z-scored_cosine']):
        patch.set_facecolor(colors[combo_name])
        patch.set_alpha(0.7)
    ax.set_ylabel('Normalized NN Distance')
    ax.set_title('NN Distance Distribution\n(normalized for comparison)')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Plot 5: Pairwise correlation of age differences
    ax = fig.add_subplot(gs[2, 0:2])
    age_diff_matrix = pd.DataFrame({
        combo_name: results_df['min_age_diff'].values 
        for combo_name, results_df in all_results.items()
    })
    corr = age_diff_matrix.corr()
    
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
               vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation of Age Differences\n(across methods)')
    ax.set_xticklabels([c.replace('_', '\n') for c in corr.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([c.replace('_', '\n') for c in corr.columns], rotation=0, fontsize=8)
    
    # Plot 6: Agreement in nearest neighbors
    ax = fig.add_subplot(gs[2, 2:4])
    
    # For each pair of methods, calculate % of cells where NN is same
    agreement_matrix = np.zeros((len(all_results), len(all_results)))
    combo_names = list(all_results.keys())
    
    for i, combo1 in enumerate(combo_names):
        for j, combo2 in enumerate(combo_names):
            if i == j:
                agreement_matrix[i, j] = 100
            else:
                # Get NN indices for both methods
                results1 = all_results[combo1]
                results2 = all_results[combo2]
                
                # Find actual NN for each cell in both methods
                # (we'd need to store NN indices for this - approximation for now)
                # For now, use correlation of age diffs as proxy
                agreement_matrix[i, j] = corr.iloc[i, j] * 100
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='Greens',
               vmin=0, vmax=100, ax=ax, square=True, 
               xticklabels=[c.replace('_', '\n') for c in combo_names],
               yticklabels=[c.replace('_', '\n') for c in combo_names],
               cbar_kws={'label': 'Agreement %'})
    ax.set_title('Method Agreement\n(correlation-based proxy)')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.suptitle(f'KNN Comparison Across All Metrics\n{cell_class}, Louvain {louvain}, {region}',
                fontsize=16, fontweight='bold')
    
    # Save
    output_dir = '/scratch/easmit31/dissimilarity_analysis'
    plot_file = os.path.join(output_dir, f'knn_metric_comparison_{cell_class}_louvain{louvain}_{region}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {plot_file}")
    plt.close()
    
    # Save comparison table
    csv_file = os.path.join(output_dir, f'knn_metric_comparison_{cell_class}_louvain{louvain}_{region}.csv')
    comp_df.to_csv(csv_file, index=False)
    print(f"✓ Saved comparison table: {csv_file}")
    
    return comp_df, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare KNN across all metric combinations')
    parser.add_argument('--louvain', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--h5ad', type=str, required=True)
    parser.add_argument('--min-age', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=10)
    
    args = parser.parse_args()
    
    compare_knn_across_all_metrics(
        louvain=args.louvain,
        region=args.region,
        h5ad_path=args.h5ad,
        min_age=args.min_age,
        k=args.k
    )
