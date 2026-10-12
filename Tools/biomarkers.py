import numpy as np
import pandas as pd
import scanpy as sc

adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)


def summarize_up_downregulated_proteins(
    adata,
    group="Cancer",
    allergen_key="byMajority",
    alpha=0.05,
    logfc_thresh=0.01
):

    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.clip(np.array(res["pvals_adj"][group]), 1e-300, None)

    is_allergen = adata.var.loc[names, allergen_key].values == "Allergen"

    # =========================================================
    # 1. ALL proteins (up/down)
    # =========================================================
    up_all = (padj < alpha) & (logfc > logfc_thresh)
    down_all = (padj < alpha) & (logfc < -logfc_thresh)

    up_all_idx = np.where(up_all)[0]
    down_all_idx = np.where(down_all)[0]

    # sort by adjusted p-value (ascending)
    up_all_idx = up_all_idx[np.argsort(padj[up_all_idx])]
    down_all_idx = down_all_idx[np.argsort(padj[down_all_idx])]

    # =========================================================
    # 2. ALLERGENS only (up/down)
    # =========================================================
    up_allergen = up_all & is_allergen
    down_allergen = down_all & is_allergen

    up_allergen_idx = np.where(up_allergen)[0]
    down_allergen_idx = np.where(down_allergen)[0]

    up_allergen_idx = up_allergen_idx[np.argsort(padj[up_allergen_idx])]
    down_allergen_idx = down_allergen_idx[np.argsort(padj[down_allergen_idx])]

    # =========================================================
    # PRINT SUMMARY
    # =========================================================
    print("\n==================== PROTEIN SUMMARY ====================")

    print("\n[ALL PROTEINS]")
    print(f"Upregulated:   {len(up_all_idx)}")
    print(f"Downregulated: {len(down_all_idx)}")

    print("\n[ALLERGEN PROTEINS]")
    print(f"Upregulated allergens:   {len(up_allergen_idx)}")
    print(f"Downregulated allergens: {len(down_allergen_idx)}")

    return {
        "up_all_idx": up_all_idx,
        "down_all_idx": down_all_idx,
        "up_allergen_idx": up_allergen_idx,
        "down_allergen_idx": down_allergen_idx,
        "names": names,
        "logfc": logfc,
        "padj": padj
    }


def export_up_downregulated_proteins(
    adata,
    group="Cancer",
    allergen_key="byMajority",
    alpha=0.05,
    logfc_thresh=0.01,
    output_dir="Biomarkers"
):
    """
    Exports:

    1. all_upregulated_proteins.csv
       Columns:
       Name, Log2FC, -Log10(FDR), Is_Allergen

    2. all_downregulated_proteins.csv
       Columns:
       Name, Log2FC, -Log10(FDR), Is_Allergen

    3. upregulated_allergens.csv
       Columns:
       Name, Log2FC, -Log10(FDR)

    4. downregulated_allergens.csv
       Columns:
       Name, Log2FC, -Log10(FDR)

    Ordering:
        Sorted by adjusted p-value ascending
        (same logic as your boxplots)
    """

    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.clip(
        np.array(res["pvals_adj"][group]),
        1e-300,
        None
    )

    neglog10_fdr = -np.log10(padj)

    is_allergen = (
        adata.var.loc[names, allergen_key]
        .eq("Allergen")
        .values
    )

    # =====================================================
    # ALL PROTEINS
    # =====================================================
    up_all = (padj < alpha) & (logfc > logfc_thresh)
    down_all = (padj < alpha) & (logfc < -logfc_thresh)

    up_all_idx = np.where(up_all)[0]
    down_all_idx = np.where(down_all)[0]

    # same ordering as your boxplots
    up_all_idx = up_all_idx[np.argsort(padj[up_all_idx])]
    down_all_idx = down_all_idx[np.argsort(padj[down_all_idx])]

    # =====================================================
    # ALLERGENS ONLY
    # =====================================================
    up_allergen = up_all & is_allergen
    down_allergen = down_all & is_allergen

    up_allergen_idx = np.where(up_allergen)[0]
    down_allergen_idx = np.where(down_allergen)[0]

    up_allergen_idx = up_allergen_idx[np.argsort(padj[up_allergen_idx])]
    down_allergen_idx = down_allergen_idx[np.argsort(padj[down_allergen_idx])]

    # =====================================================
    # DATAFRAMES
    # =====================================================

    df_up_all = pd.DataFrame({
        "Name": names[up_all_idx],
        "Log2FC": logfc[up_all_idx],
        "-Log10(FDR)": neglog10_fdr[up_all_idx],
        "Is_Allergen": is_allergen[up_all_idx]
    })

    df_down_all = pd.DataFrame({
        "Name": names[down_all_idx],
        "Log2FC": logfc[down_all_idx],
        "-Log10(FDR)": neglog10_fdr[down_all_idx],
        "Is_Allergen": is_allergen[down_all_idx]
    })

    df_up_allergen = pd.DataFrame({
        "Name": names[up_allergen_idx],
        "Log2FC": logfc[up_allergen_idx],
        "-Log10(FDR)": neglog10_fdr[up_allergen_idx]
    })

    df_down_allergen = pd.DataFrame({
        "Name": names[down_allergen_idx],
        "Log2FC": logfc[down_allergen_idx],
        "-Log10(FDR)": neglog10_fdr[down_allergen_idx]
    })

    # =====================================================
    # SAVE
    # =====================================================

    up_all_file = f"{output_dir}/all_upregulated_proteins.csv"
    down_all_file = f"{output_dir}/all_downregulated_proteins.csv"

    up_allergen_file = f"{output_dir}/upregulated_allergens.csv"
    down_allergen_file = f"{output_dir}/downregulated_allergens.csv"

    df_up_all.to_csv(up_all_file, index=False)
    df_down_all.to_csv(down_all_file, index=False)

    df_up_allergen.to_csv(up_allergen_file, index=False)
    df_down_allergen.to_csv(down_allergen_file, index=False)

    print("\n==================== PROTEIN SUMMARY ====================")
    print(f"\nUpregulated proteins:   {len(df_up_all)}")
    print(f"Downregulated proteins: {len(df_down_all)}")

    print(f"\nUpregulated allergens:   {len(df_up_allergen)}")
    print(f"Downregulated allergens: {len(df_down_allergen)}")

    print("\nSaved:")
    print(up_all_file)
    print(down_all_file)
    print(up_allergen_file)
    print(down_allergen_file)

    return {
        "up_all": df_up_all,
        "down_all": df_down_all,
        "up_allergens": df_up_allergen,
        "down_allergens": df_down_allergen
    }


export_up_downregulated_proteins(adata_ranked_proteins)
