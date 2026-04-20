import pandas as pd
import numpy as np

pca_df = pd.read_csv('pca_combined_all_celltypes.csv')

# From your output, extract variance explained
variances = [16.1, 10.5, 7.9, 1.9, 1.6, 1.2, 0.8, 0.8, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2]
total = sum(variances)

print(f"PC1-20 explain {total:.1f}% of variance")
print(f"Remaining variance: {100-total:.1f}%")
print(f"\n→ Should we go to PC50 to capture more variance?")

# Where are age signals?
age_pcs = [9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
print(f"\nAge-associated PCs: {age_pcs}")
print(f"Latest age PC: PC{max(age_pcs)}")
print(f"Could age signals exist in PC21-50? Possibly!")
