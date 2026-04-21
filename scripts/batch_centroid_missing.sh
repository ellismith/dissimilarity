#!/bin/bash
#SBATCH --job-name=centroid_missing
#SBATCH --output=/scratch/easmit31/factor_analysis/logs/centroid_missing_%A_%a.out
#SBATCH --error=/scratch/easmit31/factor_analysis/logs/centroid_missing_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-48
#SBATCH --partition=htc

mkdir -p /scratch/easmit31/factor_analysis/logs
source ~/.bashrc
conda activate latent_analysis

JOBS=(
    "GABAergic-neurons:Res1_GABAergic-neurons_subset.h5ad:lCb:"
    "GABAergic-neurons:Res1_GABAergic-neurons_subset.h5ad:MB:"
    "glutamatergic-neurons:Res1_glutamatergic-neurons_update.h5ad:lCb:"
    "glutamatergic-neurons:Res1_glutamatergic-neurons_update.h5ad:MB:"
    "basket-cells:Res1_basket-cells_update.h5ad:ACC:"
    "basket-cells:Res1_basket-cells_update.h5ad:CN:"
    "basket-cells:Res1_basket-cells_update.h5ad:dlPFC:"
    "basket-cells:Res1_basket-cells_update.h5ad:EC:"
    "basket-cells:Res1_basket-cells_update.h5ad:HIP:"
    "basket-cells:Res1_basket-cells_update.h5ad:IPP:"
    "basket-cells:Res1_basket-cells_update.h5ad:M1:"
    "basket-cells:Res1_basket-cells_update.h5ad:MB:"
    "basket-cells:Res1_basket-cells_update.h5ad:mdTN:"
    "basket-cells:Res1_basket-cells_update.h5ad:NAc:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:ACC:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:dlPFC:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:EC:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:IPP:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:lCb:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:M1:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:MB:"
    "medium-spiny-neurons:Res1_medium-spiny-neurons_subset.h5ad:mdTN:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:ACC:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:CN:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:dlPFC:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:EC:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:HIP:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:IPP:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:M1:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:MB:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:mdTN:"
    "cerebellar-neurons:Res1_cerebellar-neurons_subset.h5ad:NAc:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:ACC:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:dlPFC:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:EC:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:IPP:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:lCb:"
    "ependymal-cells:Res1_ependymal-cells_new.h5ad:M1:"
    "OPCs:Res1_opc-olig_subset.h5ad:ACC:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:CN:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:dlPFC:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:EC:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:HIP:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:IPP:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:lCb:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:M1:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:MB:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:mdTN:oligodendrocyte precursor cells"
    "OPCs:Res1_opc-olig_subset.h5ad:NAc:oligodendrocyte precursor cells"
)

ENTRY=${JOBS[$SLURM_ARRAY_TASK_ID]}
CELL_TYPE=$(echo $ENTRY | cut -d: -f1)
H5AD=$(echo $ENTRY | cut -d: -f2)
REGION=$(echo $ENTRY | cut -d: -f3)
FILTER=$(echo $ENTRY | cut -d: -f4)

echo "=========================================="
echo "Centroid analyses: $CELL_TYPE x $REGION"
echo "=========================================="

if [ -n "$FILTER" ]; then
    FILTER_ARG="--cell-class-filter $FILTER"
else
    FILTER_ARG=""
fi

python /scratch/easmit31/factor_analysis/scripts/pca_centroid_distance_by_louvain.py \
    --h5ad /data/CEM/smacklab/U01/$H5AD \
    --cell-type $CELL_TYPE \
    --region $REGION \
    $FILTER_ARG

python /scratch/easmit31/factor_analysis/scripts/pca_population_centroid_distance.py \
    --h5ad /data/CEM/smacklab/U01/$H5AD \
    --cell-type $CELL_TYPE \
    --region $REGION \
    $FILTER_ARG

echo "Done: $CELL_TYPE x $REGION"
