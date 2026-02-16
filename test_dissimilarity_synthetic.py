import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("="*70)
print("SYNTHETIC DATA TEST: Dissimilarity Matrix Validation")
print("="*70)

# =============================================================================
# TEST 1: Two distinct groups
# =============================================================================
print("\n" + "="*70)
print("TEST 1: Two Distinct Groups (Young vs Old)")
print("="*70)

n_cells_per_group = 100
n_genes = 1000

# Group 1 (Young): high expression of genes 0-499, low expression of 500-999
young_cells = np.random.poisson(lam=5, size=(n_cells_per_group, n_genes))
young_cells[:, 500:] = np.random.poisson(lam=0.5, size=(n_cells_per_group, 500))

# Group 2 (Old): low expression of genes 0-499, high expression of 500-999
old_cells = np.random.poisson(lam=0.5, size=(n_cells_per_group, n_genes))
old_cells[:, 500:] = np.random.poisson(lam=5, size=(n_cells_per_group, 500))

# Combine
X = np.vstack([young_cells, old_cells])
labels = ['young'] * n_cells_per_group + ['old'] * n_cells_per_group
ages = [5.0] * n_cells_per_group + [15.0] * n_cells_per_group

print(f"\nData shape: {X.shape}")
print(f"Group 1 (young): {n_cells_per_group} cells, mean expression = {young_cells.mean():.2f}")
print(f"Group 2 (old): {n_cells_per_group} cells, mean expression = {old_cells.mean():.2f}")

# Compute distances
dist_matrix = pairwise_distances(X, metric='euclidean')

# Within-group vs between-group distances
within_young = []
within_old = []
between_groups = []

for i in range(len(X)):
    for j in range(i+1, len(X)):
        if labels[i] == labels[j] == 'young':
            within_young.append(dist_matrix[i, j])
        elif labels[i] == labels[j] == 'old':
            within_old.append(dist_matrix[i, j])
        else:
            between_groups.append(dist_matrix[i, j])

print(f"\nDISTANCE ANALYSIS:")
print(f"  Within young: mean = {np.mean(within_young):.2f}, std = {np.std(within_young):.2f}")
print(f"  Within old:   mean = {np.mean(within_old):.2f}, std = {np.std(within_old):.2f}")
print(f"  Between groups: mean = {np.mean(between_groups):.2f}, std = {np.std(between_groups):.2f}")
print(f"  Ratio (between/within): {np.mean(between_groups) / np.mean(within_young):.2f}x")

expected_result = "Between-group distances should be MUCH larger than within-group"
actual_result = "PASS" if np.mean(between_groups) > np.mean(within_young) * 2 else "FAIL"
print(f"\n  Expected: {expected_result}")
print(f"  Result: {actual_result}")

# =============================================================================
# TEST 2: Gradual age-related change
# =============================================================================
print("\n" + "="*70)
print("TEST 2: Gradual Age-Related Change")
print("="*70)

n_cells = 200
n_genes = 1000

# Create cells with gradual age-dependent expression
# Genes 0-499: increase with age
# Genes 500-999: decrease with age
ages_gradual = np.linspace(1, 20, n_cells)
X_gradual = np.zeros((n_cells, n_genes))

for i, age in enumerate(ages_gradual):
    # Genes increase with age (0-499)
    X_gradual[i, :500] = np.random.poisson(lam=age/5, size=500)
    # Genes decrease with age (500-999)
    X_gradual[i, 500:] = np.random.poisson(lam=5 - age/5, size=500)

print(f"\nData shape: {X_gradual.shape}")
print(f"Age range: {ages_gradual.min():.1f} - {ages_gradual.max():.1f}")

# Compute distances
dist_matrix_gradual = pairwise_distances(X_gradual, metric='euclidean')

# Test: cells with similar ages should be closer
age_diffs = []
distances = []

for i in range(n_cells):
    for j in range(i+1, n_cells):
        age_diff = abs(ages_gradual[i] - ages_gradual[j])
        dist = dist_matrix_gradual[i, j]
        age_diffs.append(age_diff)
        distances.append(dist)

# Correlation
from scipy.stats import pearsonr
corr, pval = pearsonr(age_diffs, distances)

print(f"\nAGE-DISTANCE CORRELATION:")
print(f"  Correlation: {corr:.3f}")
print(f"  P-value: {pval:.2e}")

expected_result = "Positive correlation (age difference predicts distance)"
actual_result = "PASS" if corr > 0.5 and pval < 0.001 else "FAIL"
print(f"\n  Expected: {expected_result}")
print(f"  Result: {actual_result}")

# =============================================================================
# TEST 3: Random/no structure (negative control)
# =============================================================================
print("\n" + "="*70)
print("TEST 3: Random Data (Negative Control)")
print("="*70)

n_cells_random = 100
n_genes_random = 1000

# Completely random data
X_random = np.random.poisson(lam=3, size=(n_cells_random, n_genes_random))
ages_random = np.random.uniform(1, 20, n_cells_random)

dist_matrix_random = pairwise_distances(X_random, metric='euclidean')

# Test: age should NOT predict distance
age_diffs_random = []
distances_random = []

for i in range(n_cells_random):
    for j in range(i+1, n_cells_random):
        age_diff = abs(ages_random[i] - ages_random[j])
        dist = dist_matrix_random[i, j]
        age_diffs_random.append(age_diff)
        distances_random.append(dist)

corr_random, pval_random = pearsonr(age_diffs_random, distances_random)

print(f"\nAGE-DISTANCE CORRELATION:")
print(f"  Correlation: {corr_random:.3f}")
print(f"  P-value: {pval_random:.2f}")

expected_result = "No correlation (random data)"
actual_result = "PASS" if abs(corr_random) < 0.1 else "FAIL"
print(f"\n  Expected: {expected_result}")
print(f"  Result: {actual_result}")

# =============================================================================
# TEST 4: Known nearest neighbors
# =============================================================================
print("\n" + "="*70)
print("TEST 4: Known Nearest Neighbors")
print("="*70)

# Create 5 clusters, each with 20 cells
n_clusters = 5
cells_per_cluster = 20
n_genes_cluster = 500

X_clusters = []
cluster_labels = []
cluster_ages = []

for c in range(n_clusters):
    # Each cluster has a different mean expression profile
    cluster_mean = np.random.uniform(1, 10, n_genes_cluster)
    
    for i in range(cells_per_cluster):
        # Cells within cluster are similar
        cell = np.random.poisson(lam=cluster_mean)
        X_clusters.append(cell)
        cluster_labels.append(c)
        # Assign ages randomly within cluster
        cluster_ages.append(np.random.uniform(5, 15))

X_clusters = np.array(X_clusters)
cluster_labels = np.array(cluster_labels)
cluster_ages = np.array(cluster_ages)

print(f"\nData: {n_clusters} clusters, {cells_per_cluster} cells each")

dist_matrix_clusters = pairwise_distances(X_clusters, metric='euclidean')

# For each cell, find nearest neighbor
correct_nn = 0
total_cells = len(X_clusters)

for i in range(total_cells):
    distances = dist_matrix_clusters[i, :].copy()
    distances[i] = np.inf  # Exclude self
    nn_idx = np.argmin(distances)
    
    # Is nearest neighbor from same cluster?
    if cluster_labels[i] == cluster_labels[nn_idx]:
        correct_nn += 1

pct_correct = (correct_nn / total_cells) * 100

print(f"\nNEAREST NEIGHBOR ANALYSIS:")
print(f"  Nearest neighbors from same cluster: {correct_nn}/{total_cells} ({pct_correct:.1f}%)")

expected_result = "Most (>80%) nearest neighbors should be from same cluster"
actual_result = "PASS" if pct_correct > 80 else "FAIL"
print(f"\n  Expected: {expected_result}")
print(f"  Result: {actual_result}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("Creating Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Test 1: Two groups heatmap
ax = axes[0, 0]
im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')
ax.axhline(n_cells_per_group, color='red', linewidth=2)
ax.axvline(n_cells_per_group, color='red', linewidth=2)
ax.set_title('Test 1: Two Distinct Groups\n(Young vs Old)')
ax.set_xlabel('Cell index')
ax.set_ylabel('Cell index')
plt.colorbar(im, ax=ax)

# Test 1: Distance distributions
ax = axes[0, 1]
ax.hist([within_young, between_groups], bins=30, 
        label=['Within group', 'Between groups'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Distance')
ax.set_ylabel('Count')
ax.set_title('Test 1: Distance Distributions')
ax.legend()

# Test 2: Age vs distance scatter
ax = axes[0, 2]
ax.scatter(age_diffs, distances, alpha=0.3, s=1)
ax.set_xlabel('Age difference (years)')
ax.set_ylabel('Distance')
ax.set_title(f'Test 2: Age-Distance Correlation\nr={corr:.3f}')

# Test 2: Heatmap
ax = axes[1, 0]
im = ax.imshow(dist_matrix_gradual, cmap='viridis', aspect='auto')
ax.set_title('Test 2: Gradual Age Change')
ax.set_xlabel('Cell index (ordered by age)')
ax.set_ylabel('Cell index (ordered by age)')
plt.colorbar(im, ax=ax)

# Test 3: Random scatter
ax = axes[1, 1]
ax.scatter(age_diffs_random, distances_random, alpha=0.3, s=1)
ax.set_xlabel('Age difference (years)')
ax.set_ylabel('Distance')
ax.set_title(f'Test 3: Random Data (Negative Control)\nr={corr_random:.3f}')

# Test 4: Clusters heatmap
ax = axes[1, 2]
im = ax.imshow(dist_matrix_clusters, cmap='viridis', aspect='auto')
# Add lines between clusters
for c in range(1, n_clusters):
    ax.axhline(c * cells_per_cluster, color='red', linewidth=1)
    ax.axvline(c * cells_per_cluster, color='red', linewidth=1)
ax.set_title(f'Test 4: 5 Clusters\n{pct_correct:.1f}% correct NN')
ax.set_xlabel('Cell index')
ax.set_ylabel('Cell index')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/scratch/easmit31/dissimilarity_analysis/synthetic_test_results.png', 
            dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: synthetic_test_results.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_pass = all([
    np.mean(between_groups) > np.mean(within_young) * 2,
    corr > 0.5 and pval < 0.001,
    abs(corr_random) < 0.1,
    pct_correct > 80
])

if all_pass:
    print("\n✓ ALL TESTS PASSED!")
    print("  The dissimilarity calculations are working correctly.")
    print("  You can trust these methods for your real data.")
else:
    print("\n✗ SOME TESTS FAILED")
    print("  Check the results above to see which tests failed.")

print("="*70)
