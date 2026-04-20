#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CREATING REGION MARKER VISUALIZATIONS (η² > 0.1)")
print("="*70)

# Load component lists
pca_comps = pd.read_csv('pca_region_components_0.1.csv')
fa_comps = pd.read_csv('fa_region_components_0.1.csv')

print(f"\nVisualizing {len(pca_comps)} PCA and {len(fa_comps)} FA components")

# ========== 1. GRID OF BAR PLOTS - PCA ==========
print("\nCreating PCA region marker grids...")

# Calculate grid size
n_comps = len(pca_comps)
n_cols = 4
n_rows = int(np.ceil(n_comps / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for idx, (_, row) in enumerate(pca_comps.iterrows()):
    comp = row['Component']
    ax = axes[idx]
    
    # Load marker genes
    df = pd.read_csv(f'pca_{comp}_region_markers.csv')
    
    # Get top 10 from each direction
    pos = df[df['direction'] == 'positive'].head(10)
    neg = df[df['direction'] == 'negative'].head(10)
    
    # Combine and sort
    combined = pd.concat([pos, neg])
    combined['abs_loading'] = combined['loading'].abs()
    combined = combined.sort_values('abs_loading', ascending=True)
    
    # Plot
    colors = ['red' if x > 0 else 'blue' for x in combined['loading']]
    ax.barh(range(len(combined)), combined['loading'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['gene'], fontsize=6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Loading', fontsize=8)
    ax.set_title(f"{comp}: {row['Dominant_Region']} (η²={row['Region_eta2']:.2f})", 
                 fontweight='bold', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # Add region labels
    if len(pos) > 0 and len(neg) > 0:
        pos_regs = pos.iloc[0]['regions']
        neg_regs = neg.iloc[0]['regions']
        ax.text(0.02, 0.98, f'{pos_regs}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
        ax.text(0.02, 0.88, f'{neg_regs}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Hide unused subplots
for idx in range(n_comps, len(axes)):
    axes[idx].axis('off')

plt.suptitle('PCA: Region Marker Genes (η² > 0.1)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('pca_all_region_markers_grid.png', dpi=300, bbox_inches='tight')
print("Saved: pca_all_region_markers_grid.png")
plt.close()

# ========== 2. GRID OF BAR PLOTS - FA ==========
print("\nCreating FA region marker grids...")

n_comps = len(fa_comps)
n_rows = int(np.ceil(n_comps / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for idx, (_, row) in enumerate(fa_comps.iterrows()):
    comp = row['Component']
    ax = axes[idx]
    
    # Load marker genes
    df = pd.read_csv(f'fa_{comp}_region_markers.csv')
    
    # Get top 10 from each direction
    pos = df[df['direction'] == 'positive'].head(10)
    neg = df[df['direction'] == 'negative'].head(10)
    
    # Combine and sort
    combined = pd.concat([pos, neg])
    combined['abs_loading'] = combined['loading'].abs()
    combined = combined.sort_values('abs_loading', ascending=True)
    
    # Plot
    colors = ['red' if x > 0 else 'blue' for x in combined['loading']]
    ax.barh(range(len(combined)), combined['loading'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['gene'], fontsize=6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Loading', fontsize=8)
    ax.set_title(f"{comp}: {row['Dominant_Region']} (η²={row['Region_eta2']:.2f})", 
                 fontweight='bold', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # Add region labels
    if len(pos) > 0 and len(neg) > 0:
        pos_regs = pos.iloc[0]['regions']
        neg_regs = neg.iloc[0]['regions']
        ax.text(0.02, 0.98, f'{pos_regs}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
        ax.text(0.02, 0.88, f'{neg_regs}', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Hide unused subplots
for idx in range(n_comps, len(axes)):
    axes[idx].axis('off')

plt.suptitle('FA: Region Marker Genes (η² > 0.1)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('fa_all_region_markers_grid.png', dpi=300, bbox_inches='tight')
print("Saved: fa_all_region_markers_grid.png")
plt.close()

# ========== 3. SUMMARY HEATMAP - REGION FREQUENCY ==========
print("\nCreating region frequency summary...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
ax = axes[0]
region_counts = pca_comps['Dominant_Region'].value_counts()
regions_sorted = region_counts.index.tolist()
counts = region_counts.values

ax.barh(range(len(regions_sorted)), counts, color='steelblue', alpha=0.7)
ax.set_yticks(range(len(regions_sorted)))
ax.set_yticklabels(regions_sorted)
ax.set_xlabel('Number of Components', fontweight='bold')
ax.set_title('PCA: Components per Region (η² > 0.1)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add counts on bars
for i, count in enumerate(counts):
    ax.text(count + 0.1, i, str(count), va='center', fontsize=9)

# FA
ax = axes[1]
region_counts = fa_comps['Dominant_Region'].value_counts()
regions_sorted = region_counts.index.tolist()
counts = region_counts.values

ax.barh(range(len(regions_sorted)), counts, color='coral', alpha=0.7)
ax.set_yticks(range(len(regions_sorted)))
ax.set_yticklabels(regions_sorted)
ax.set_xlabel('Number of Factors', fontweight='bold')
ax.set_title('FA: Factors per Region (η² > 0.1)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add counts on bars
for i, count in enumerate(counts):
    ax.text(count + 0.1, i, str(count), va='center', fontsize=9)

plt.suptitle('How Many Components Represent Each Region?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('region_component_frequency.png', dpi=300, bbox_inches='tight')
print("Saved: region_component_frequency.png")
plt.close()

# ========== 4. EFFECT SIZE DISTRIBUTION ==========
print("\nCreating effect size distributions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
ax = axes[0]
eta2_vals = pca_comps['Region_eta2'].values
comps = pca_comps['Component'].values

colors = ['darkblue' if x > 0.3 else 'steelblue' for x in eta2_vals]
ax.barh(range(len(comps)), eta2_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(comps)))
ax.set_yticklabels(comps, fontsize=8)
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Strong threshold (0.3)')
ax.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='Weak threshold (0.1)')
ax.set_xlabel('Region η²', fontweight='bold')
ax.set_title('PCA: Regional Effect Sizes', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)

# FA
ax = axes[1]
eta2_vals = fa_comps['Region_eta2'].values
comps = fa_comps['Component'].values

colors = ['darkred' if x > 0.3 else 'coral' for x in eta2_vals]
ax.barh(range(len(comps)), eta2_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(comps)))
ax.set_yticklabels(comps, fontsize=8)
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Strong threshold (0.3)')
ax.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='Weak threshold (0.1)')
ax.set_xlabel('Region η²', fontweight='bold')
ax.set_title('FA: Regional Effect Sizes', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Regional Effect Sizes Across Components', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('region_effect_sizes.png', dpi=300, bbox_inches='tight')
print("Saved: region_effect_sizes.png")
plt.close()

print("\n" + "="*70)
print("Complete!")
print("="*70)
print("\nCreated:")
print("  1. pca_all_region_markers_grid.png - All PCA region marker genes")
print("  2. fa_all_region_markers_grid.png - All FA region marker genes")
print("  3. region_component_frequency.png - How many components per region")
print("  4. region_effect_sizes.png - Effect size distribution")

