import scanpy as sc
import matplotlib.pyplot as plt
from os.path import exists
from os.path import join
import pandas as pd
import scanpy as sc
import numpy as np
import anndata
import json

plt.rcParams['font.family'] = 'Times New Roman'


def rank_proteins(adata):
    sc.tl.rank_genes_groups(adata, groupby='label', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,show=False)

    adata.uns["rank_proteins_groups"] = adata.uns["rank_genes_groups"]
    sc.write("../H5AD/adata_ranked_proteins.h5ad", adata)

    fig = plt.gcf()
    # Option 2
    fig.set_size_inches(18, 7)
    plt.subplots_adjust(left=0.55, right=0.98, top=0.95, bottom=0.15, wspace=0.4)
    plt.savefig("Figures/rank_proteins2.png", dpi=1500, bbox_inches="tight")

    plt.close()

