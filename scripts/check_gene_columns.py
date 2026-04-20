#!/usr/bin/env python

import scanpy as sc

print("Checking what gene info is available...\n")

adata = sc.read_h5ad('/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/Res1_glutamatergic-neurons_update.h5ad', backed='r')

print("adata.var columns:")
print(adata.var.columns.tolist())

print("\nFirst 10 rows of adata.var:")
print(adata.var.head(10))

print("\nadata.var_names (gene IDs):")
print(adata.var_names[:10].tolist())

