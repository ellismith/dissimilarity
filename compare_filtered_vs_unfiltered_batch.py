import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def compare_all_filtered_vs_unfiltered(cell_type='GABAergic-neurons'):
    """
    Compare OLD (filtered - exclude same-animal) vs NEW (unfiltered - allow same-animal)
    across all louvain-region combinations
    """
    
    base_dir = f"/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices/{cell_type}"
    
    print(f"Comparing filtered vs unfiltered for {cell_type}")
    print(f"{'='*70}\n")
    
    # Find all old filtered files
    old_files = glob.glob(os.path.join(base_dir, "*_validation_summary_old.csv"))
    
    # Find all new unfiltered files
    new_files = glob.glob(os.path.join(base_dir, "*_validation_summary_no_animal_filter.csv"))
    
    print(f"Found {len(old_files)} old filtered files")
    print(f"Found {len(new_files)} new unfiltered files")
    
    # Match them up
    comparisons = []
    
    for new_file in new_files:
        # Extract base name
        basename = os.path.basename(new_file).replace('_validation_summary_no_animal_filter.csv', '')
        
        # Find corresponding old file
        # Old files have pattern: louvain{X}_{region}_minage{Y}_validation_summary_old.csv
        old_file = os.path.join(base_dir, f"{basename}_validation_summary_old.csv")
        
        if not os.path.exists(old_file):
            print(f"  Warning: No old file for {basename}")
            continue
        
        # Load both
        old_val = pd.read_csv(old_file).iloc[0]
        new_val = pd.read_csv(new_file).iloc[0]
        
        # Extract louvain and region
        parts = basename.split('_')
        louvain = parts[0].replace('louvain', '')
        region = parts[1]
        
        # Compare
        comparison = {
            'louvain': louvain,
            'region': region,
            'basename': basename,
            'n_cells': new_val['n_cells'],
            'n_animals': new_val['n_animals'],
            
            # Animal clustering
            'old_pct_diff_animal': old_val['pct_diff_animal_neighbors'],
            'new_pct_diff_animal': new_val['pct_diff_animal_neighbors'],
            'new_pct_same_animal': new_val['pct_same_animal_neighbors'],
            
            # Animal effects
            'old_ratio_diff_same': old_val['ratio_diff_to_same'],
            'new_ratio_diff_same': new_val['ratio_diff_to_same'],
            'delta_ratio_diff_same': new_val['ratio_diff_to_same'] - old_val['ratio_diff_to_same'],
            
            # Structure
            'old_ratio_random_nn': old_val['ratio_random_to_nn'],
            'new_ratio_random_nn': new_val['ratio_random_to_nn'],
            'delta_ratio_random_nn': new_val['ratio_random_to_nn'] - old_val['ratio_random_to_nn'],
            
            # Age clustering
            'old_mean_age_diff_nn': old_val.get('mean_age_diff_nn', np.nan),
            'new_mean_age_diff_nn': new_val.get('mean_age_diff_nn', np.nan),
            'delta_mean_age_diff_nn': new_val.get('mean_age_diff_nn', np.nan) - old_val.get('mean_age_diff_nn', np.nan)
        }
        
        comparisons.append(comparison)
    
    comp_df = pd.DataFrame(comparisons)
    comp_df = comp_df.sort_values(['region', 'louvain'])
    
    print(f"\nSuccessfully compared {len(comp_df)} combinations\n")
    
    # Summary statistics
    print(f"{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    print("Same-animal neighbors (new approach):")
    print(f"  Mean: {comp_df['new_pct_same_animal'].mean():.2f}%")
    print(f"  Min: {comp_df['new_pct_same_animal'].min():.2f}%")
    print(f"  Max: {comp_df['new_pct_same_animal'].max():.2f}%")
    
    print(f"\nAnimal effect ratio change:")
    print(f"  Mean delta: {comp_df['delta_ratio_diff_same'].mean():+.3f}")
    print(f"  Improved (more <1.0): {(comp_df['delta_ratio_diff_same'] < 0).sum()}/{len(comp_df)}")
    
    print(f"\nStructure (random/NN ratio) change:")
    print(f"  Mean delta: {comp_df['delta_ratio_random_nn'].mean():+.3f}")
    print(f"  Improved (higher): {(comp_df['delta_ratio_random_nn'] > 0).sum()}/{len(comp_df)}")
    
    print(f"\nAge clustering change:")
    print(f"  Mean delta: {comp_df['delta_mean_age_diff_nn'].mean():+.2f} years")
    print(f"  Age diff increased: {(comp_df['delta_mean_age_diff_nn'] > 1).sum()}/{len(comp_df)}")
    
    # By region
    print(f"\n{'='*70}")
    print("BY REGION")
    print(f"{'='*70}\n")
    
    for region in comp_df['region'].unique():
        region_df = comp_df[comp_df['region'] == region]
        print(f"{region}:")
        print(f"  Combinations: {len(region_df)}")
        print(f"  Mean same-animal neighbors: {region_df['new_pct_same_animal'].mean():.2f}%")
        print(f"  Mean age diff change: {region_df['delta_mean_age_diff_nn'].mean():+.2f} years")
        print()
    
    # Save comparison
    output_file = os.path.join(base_dir, 'filtered_vs_unfiltered_comparison_all.csv')
    comp_df.to_csv(output_file, index=False)
    print(f"✓ Saved comparison: {output_file}\n")
    
    # Create comprehensive visualization
    print(f"{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}\n")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Same-animal neighbors distribution
    ax = fig.add_subplot(gs[0, 0:2])
    ax.hist(comp_df['new_pct_same_animal'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(comp_df['new_pct_same_animal'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f"Mean={comp_df['new_pct_same_animal'].mean():.1f}%")
    ax.set_xlabel('% Same-Animal Neighbors (New Approach)')
    ax.set_ylabel('Count')
    ax.set_title('Same-Animal Clustering\n(When Allowed)')
    ax.legend()
    
    # Plot 2: Same-animal neighbors by region
    ax = fig.add_subplot(gs[0, 2:4])
    regions = comp_df['region'].unique()
    region_data = [comp_df[comp_df['region'] == r]['new_pct_same_animal'].values for r in regions]
    bp = ax.boxplot(region_data, labels=regions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(10, color='red', linestyle='--', alpha=0.3, label='10% threshold')
    ax.set_ylabel('% Same-Animal Neighbors')
    ax.set_title('Same-Animal Clustering by Region')
    ax.legend()
    
    # Plot 3: Animal effect ratio - old vs new
    ax = fig.add_subplot(gs[1, 0:2])
    ax.scatter(comp_df['old_ratio_diff_same'], comp_df['new_ratio_diff_same'], 
              s=50, alpha=0.6, edgecolor='black')
    lim = max(comp_df['old_ratio_diff_same'].max(), comp_df['new_ratio_diff_same'].max()) + 0.05
    ax.plot([1, lim], [1, lim], 'r--', alpha=0.5, label='Equal')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Old (Filtered) - Ratio diff/same')
    ax.set_ylabel('New (Unfiltered) - Ratio diff/same')
    ax.set_title('Animal Effect Ratio: Old vs New')
    ax.legend()
    
    # Plot 4: Animal effect delta
    ax = fig.add_subplot(gs[1, 2:4])
    colors = ['green' if d < 0 else 'red' for d in comp_df['delta_ratio_diff_same']]
    ax.barh(range(len(comp_df)), comp_df['delta_ratio_diff_same'], color=colors, alpha=0.6)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Change in Ratio (negative = improved)')
    ax.set_ylabel('Combination Index')
    ax.set_title('Animal Effect Change\n(Green = Improved)')
    
    # Plot 5: Structure ratio - old vs new
    ax = fig.add_subplot(gs[2, 0:2])
    ax.scatter(comp_df['old_ratio_random_nn'], comp_df['new_ratio_random_nn'],
              s=50, alpha=0.6, edgecolor='black')
    lim_min = min(comp_df['old_ratio_random_nn'].min(), comp_df['new_ratio_random_nn'].min()) - 0.1
    lim_max = max(comp_df['old_ratio_random_nn'].max(), comp_df['new_ratio_random_nn'].max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='Equal')
    ax.set_xlabel('Old (Filtered) - Random/NN Ratio')
    ax.set_ylabel('New (Unfiltered) - Random/NN Ratio')
    ax.set_title('Structure Quality: Old vs New')
    ax.legend()
    
    # Plot 6: Structure delta
    ax = fig.add_subplot(gs[2, 2:4])
    ax.hist(comp_df['delta_ratio_random_nn'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(comp_df['delta_ratio_random_nn'].mean(), color='blue', linestyle='--',
              linewidth=2, label=f"Mean={comp_df['delta_ratio_random_nn'].mean():+.3f}")
    ax.set_xlabel('Change in Structure Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Structure Change Distribution\n(Positive = Improved)')
    ax.legend()
    
    # Plot 7: Age difference - old vs new
    ax = fig.add_subplot(gs[3, 0:2])
    valid_age = comp_df.dropna(subset=['old_mean_age_diff_nn', 'new_mean_age_diff_nn'])
    ax.scatter(valid_age['old_mean_age_diff_nn'], valid_age['new_mean_age_diff_nn'],
              s=50, alpha=0.6, edgecolor='black', c=valid_age['new_pct_same_animal'],
              cmap='RdYlGn_r', vmin=0, vmax=15)
    lim_max = max(valid_age['old_mean_age_diff_nn'].max(), valid_age['new_mean_age_diff_nn'].max()) + 0.5
    ax.plot([0, lim_max], [0, lim_max], 'r--', alpha=0.5, label='Equal')
    ax.set_xlabel('Old (Filtered) - Age Diff to NN (years)')
    ax.set_ylabel('New (Unfiltered) - Age Diff to NN (years)')
    ax.set_title('Age Clustering: Old vs New\n(Color = % same-animal neighbors)')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('% Same-Animal')
    ax.legend()
    
    # Plot 8: Age difference delta
    ax = fig.add_subplot(gs[3, 2:4])
    valid_age_delta = comp_df.dropna(subset=['delta_mean_age_diff_nn'])
    ax.hist(valid_age_delta['delta_mean_age_diff_nn'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(valid_age_delta['delta_mean_age_diff_nn'].mean(), color='blue', linestyle='--',
              linewidth=2, label=f"Mean={valid_age_delta['delta_mean_age_diff_nn'].mean():+.2f}y")
    ax.set_xlabel('Change in Age Diff to NN (years)')
    ax.set_ylabel('Count')
    ax.set_title('Age Clustering Change\n(Positive = weaker clustering)')
    ax.legend()
    
    plt.suptitle(f'Filtered vs Unfiltered Comparison: {cell_type}\n' + 
                f'{len(comp_df)} combinations', fontsize=16, fontweight='bold')
    
    plot_file = os.path.join(base_dir, 'filtered_vs_unfiltered_comparison_all.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {plot_file}")
    plt.close()
    
    return comp_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare filtered vs unfiltered across all combinations')
    parser.add_argument('--cell-type', type=str, default='GABAergic-neurons')
    
    args = parser.parse_args()
    
    compare_all_filtered_vs_unfiltered(args.cell_type)
