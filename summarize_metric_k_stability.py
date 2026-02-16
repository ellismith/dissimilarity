import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Find all stability result files
files = glob.glob('/scratch/easmit31/dissimilarity_analysis/metric_k_stability_*.csv')

print(f"Found {len(files)} stability analysis files")

all_data = []
for file in files:
    df = pd.read_csv(file)
    # Extract louvain from filename
    louvain = file.split('louvain')[1].split('_')[0]
    df['louvain'] = louvain
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

print(f"\nTotal comparisons: {len(combined)}")
print(f"Louvains: {sorted(combined['louvain'].unique())}")

# Summary statistics
print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

# Same metric, different k
same_metric_diff_k = combined[(combined['same_metric'] == True) & (combined['same_k'] == False)]
print(f"\nSame metric, different k:")
print(f"  Mean overlap: {same_metric_diff_k['jaccard'].mean():.1f}%")

# Different metrics, same k
diff_metric_same_k = combined[(combined['same_metric'] == False) & (combined['same_k'] == True)]
print(f"\nDifferent metrics, same k:")
print(f"  Mean overlap: {diff_metric_same_k['jaccard'].mean():.1f}%")

# Key comparisons at k=10
k10 = combined[(combined['k1'] == 10) & (combined['k2'] == 10)]

print(f"\n" + "="*70)
print("KEY COMPARISONS (k=10)")
print("="*70)

comparisons = [
    ('Raw_Euclidean', 'Raw_Cosine', 'Euc vs Cos (Raw)'),
    ('Raw_Euclidean', 'Raw_Correlation', 'Euc vs Corr (Raw)'),
    ('Raw_Cosine', 'Raw_Correlation', 'Cos vs Corr (Raw)'),
    ('Raw_Euclidean', 'Zscore_Euclidean', 'Raw vs Zscore (Euc)'),
]

for m1, m2, name in comparisons:
    subset = k10[
        (((k10['metric1'] == m1) & (k10['metric2'] == m2)) |
         ((k10['metric1'] == m2) & (k10['metric2'] == m1)))
    ]
    print(f"\n{name}:")
    print(f"  Mean: {subset['jaccard'].mean():.1f}%")
    print(f"  Range: {subset['jaccard'].min():.1f}% - {subset['jaccard'].max():.1f}%")

# Save summary
summary_file = '/scratch/easmit31/dissimilarity_analysis/metric_k_stability_summary.csv'
combined.to_csv(summary_file, index=False)
print(f"\n✓ Saved combined results: {summary_file}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Euc vs Cos by louvain
ax = axes[0, 0]
euc_cos = k10[
    (((k10['metric1'] == 'Raw_Euclidean') & (k10['metric2'] == 'Raw_Cosine')) |
     ((k10['metric1'] == 'Raw_Cosine') & (k10['metric2'] == 'Raw_Euclidean')))
]
euc_cos_summary = euc_cos.groupby('louvain')['jaccard'].mean().reset_index()
ax.bar(range(len(euc_cos_summary)), euc_cos_summary['jaccard'], edgecolor='black')
ax.set_xticks(range(len(euc_cos_summary)))
ax.set_xticklabels(euc_cos_summary['louvain'], rotation=45)
ax.set_xlabel('Louvain')
ax.set_ylabel('Neighbor Overlap (%)')
ax.set_title('Euclidean vs Cosine (k=10)\nAcross Louvains')
ax.axhline(euc_cos['jaccard'].mean(), color='red', linestyle='--', label=f"Mean={euc_cos['jaccard'].mean():.1f}%")
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Distribution of all key comparisons
ax = axes[0, 1]
comparison_data = []
comparison_labels = []

for m1, m2, name in comparisons:
    subset = k10[
        (((k10['metric1'] == m1) & (k10['metric2'] == m2)) |
         ((k10['metric1'] == m2) & (k10['metric2'] == m1)))
    ]
    if len(subset) > 0:
        comparison_data.append(subset['jaccard'].values)
        comparison_labels.append(name)

bp = ax.boxplot(comparison_data, labels=comparison_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Neighbor Overlap (%)')
ax.set_title('Metric Comparisons at k=10\n(All Louvains Combined)')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

# Plot 3: Effect of k (averaged)
ax = axes[1, 0]
same_k_summary = same_metric_diff_k.copy()
same_k_summary['k_diff'] = np.abs(same_k_summary['k1'] - same_k_summary['k2'])
k_effect = same_k_summary.groupby('k_diff')['jaccard'].agg(['mean', 'std']).reset_index()

ax.errorbar(k_effect['k_diff'], k_effect['mean'], yerr=k_effect['std'], 
           marker='o', linewidth=2, capsize=5, label='Mean ± SD')
ax.set_xlabel('Difference in k values')
ax.set_ylabel('Mean Neighbor Overlap (%)')
ax.set_title('Effect of k Value\n(Same metric, all metrics & louvains)')
ax.grid(alpha=0.3)
ax.legend()

# Plot 4: Summary bar
ax = axes[1, 1]
summary_stats = {
    'Same metric\ndiff k': same_metric_diff_k['jaccard'].mean(),
    'Diff metrics\nsame k': diff_metric_same_k['jaccard'].mean(),
    'Euc vs Cos\n(k=10)': euc_cos['jaccard'].mean(),
    'Cos vs Corr\n(k=10)': k10[
        (((k10['metric1'] == 'Raw_Cosine') & (k10['metric2'] == 'Raw_Correlation')) |
         ((k10['metric1'] == 'Raw_Correlation') & (k10['metric2'] == 'Raw_Cosine')))
    ]['jaccard'].mean(),
}

bars = ax.bar(range(len(summary_stats)), list(summary_stats.values()), 
             color=['green', 'orange', 'blue', 'purple'], alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(summary_stats)))
ax.set_xticklabels(list(summary_stats.keys()), fontsize=10)
ax.set_ylabel('Mean Neighbor Overlap (%)')
ax.set_title('Summary Statistics\n(All Louvains)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, summary_stats.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
           ha='center', va='bottom', fontweight='bold')

plt.suptitle('Metric & K-Value Stability: Summary Across All Louvains', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

plot_file = '/scratch/easmit31/dissimilarity_analysis/metric_k_stability_summary.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved summary plot: {plot_file}")

