#!/bin/bash

conda create -n latent_analysis python=3.10 -y
conda activate latent_analysis
conda install -c conda-forge scanpy numpy pandas scipy scikit-learn matplotlib seaborn -y
pip install anndata
