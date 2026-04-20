#!/usr/bin/env python

import scanpy as sc
import pandas as pd
import numpy as np

# Load in backed mode to conserve memory
print("Loading data...")
adata = sc.read_h5ad(
    '/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_astrocytes_update.h5ad',
    backed='r'
)

print("Data loaded successfully!")

# Check the structure
print("\nShape:", adata.shape)
print("\n=== Available metadata (obs columns) ===")
print(adata.obs.columns.tolist())

# Look at first few rows of metadata
print("\n=== First few rows of metadata ===")
print(adata.obs.head())

# Check unique values for key variables
if 'animal' in adata.obs.columns:
    print("\n=== Unique animals ===")
    print(adata.obs['animal'].value_counts())
else:
    print("\nNo 'animal' column found")

if 'age' in adata.obs.columns:
    print("\n=== Age distribution ===")
    print(adata.obs['age'].value_counts())
else:
    print("\nNo 'age' column found")

if 'region' in adata.obs.columns:
    print("\n=== Regions ===")
    print(adata.obs['region'].value_counts())
else:
    print("\nNo 'region' column found")

if 'sex' in adata.obs.columns:
    print("\n=== Sex ===")
    print(adata.obs['sex'].value_counts())
else:
    print("\nNo 'sex' column found")
