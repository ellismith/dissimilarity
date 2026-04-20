import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path

# directory with your CSVs
data_dir = Path("csv_files")   # your CSV folder

results = []

for f in sorted(data_dir.glob("pca_analysis_*_*.csv")):
    df = pd.read_csv(f)

    # select numeric columns only (skip metadata)
    numeric_cols = df.select_dtypes(include='number').columns
    X = df[numeric_cols]

    if X.empty:
        print(f"Skipping {f.name} (no numeric columns)")
        continue

    pca = PCA()
    pca.fit(X)

    ev = pca.explained_variance_ratio_

    results.append({
        "file": f.name,
        "explained_variance": ev.tolist()
    })

# print results
for r in results:
    print(r["file"])
    print("  explained variance:", r["explained_variance"])
    print()
