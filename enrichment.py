import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact


adata_ranked_proteins_path = "H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)


def report_allergen_enrichment(
    adata,
    group="Cancer",
    allergen_col="byMajority",
    alpha=0.05,
    logfc_thresh=0.01,
    mode="all"  # "all", "up", "down"
):

    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.clip(np.array(res["pvals_adj"][group]), 1e-300, None)

    is_allergen = adata.var.loc[names, allergen_col].values == "Allergen"

    # =====================================================
    # DEFINE SIGNIFICANCE MASKS (consistent with your pipeline)
    # =====================================================
    if mode == "all":
        sig_mask = (padj < alpha) & (np.abs(logfc) > logfc_thresh)

    elif mode == "up":
        sig_mask = (padj < alpha) & (logfc > logfc_thresh)

    elif mode == "down":
        sig_mask = (padj < alpha) & (logfc < -logfc_thresh)

    else:
        raise ValueError("mode must be one of: 'all', 'up', 'down'")

    # =====================================================
    # CONTINGENCY TABLE
    # =====================================================
    a = np.sum(sig_mask & is_allergen)
    b = np.sum(~sig_mask & is_allergen)
    c = np.sum(sig_mask & ~is_allergen)
    d = np.sum(~sig_mask & ~is_allergen)

    table = np.array([[a, b],
                      [c, d]])

    oddsratio, pvalue = fisher_exact(table, alternative="greater")

    # =====================================================
    # OUTPUT
    # =====================================================
    print("\n==================== ALLERGEN ENRICHMENT ====================")
    print(f"Mode: {mode}")
    print(pd.DataFrame(
        table,
        index=["Allergen", "Non-Allergen"],
        columns=["Significant", "Not Significant"]
    ))

    print(f"\nOdds ratio: {oddsratio:.3f}")
    print(f"P-value: {pvalue:.3e}")

    return {
        "table": table,
        "oddsratio": oddsratio,
        "pvalue": pvalue
    }

# all
# Are allergens enriched among any differentially expressed proteins?
# Are allergen proteins generally more involved in disease-related perturbation?

# up
# Are allergens enriched among proteins upregulated in cancer?
# Are allergens more likely to increase in cancer?

# down
# Are allergens enriched among proteins downregulated in cancer?
# Are allergens more likely to decrease in cancer?


print(report_allergen_enrichment(adata_ranked_proteins, mode="all"))
print(report_allergen_enrichment(adata_ranked_proteins, mode="up"))
print(report_allergen_enrichment(adata_ranked_proteins, mode="down"))
