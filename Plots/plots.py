import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from os.path import exists
from os.path import join
import pandas as pd
import anndata
import json
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl


# font_path = "Temp/times.ttf"
# fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
# print([f.name for f in fm.fontManager.ttflist if "Times" in f.name])
# print(plt.rcParams['font.size'])


def plot_allergen_expression(adata, point_size=60, random_state=0, use_zscore=True, percentile_clip=(1, 99)):
    allergen_columns = [
        "AlgPred2",
        "DeepAlgPro",
        "AllergenAI",
        "byAND",
        "byOR",
        "byMajority"
    ]

    adata_umap = adata.copy()

    # -----------------------
    # Expression matrix
    # -----------------------
    expr = adata_umap.X

    if hasattr(expr, "toarray"):
        expr = expr.toarray()

    # -----------------------
    # UMAP
    # -----------------------
    sc.pp.neighbors(adata_umap, use_rep="X")
    sc.tl.umap(adata_umap, random_state=random_state)

    # -----------------------
    # Compute scores
    # -----------------------
    all_scores = []

    for col in allergen_columns:

        mask = (
            adata.var[col]
            .astype(str)
            .eq("Allergen")
            .values
        )

        cell_score = expr[:, mask].sum(axis=1)

        if use_zscore:
            cell_score = (
                cell_score - cell_score.mean()
            ) / (cell_score.std() + 1e-9)

        adata_umap.obs[f"{col}_allergen_score"] = cell_score

        all_scores.extend(cell_score)

    # -----------------------
    # Shared color scale
    # -----------------------
    all_scores = np.array(all_scores)

    vmin, vmax = np.percentile(
        all_scores,
        percentile_clip
    )

    # OPTIONAL:
    # make symmetric around 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs

    # Example:
    # forces approximately [-2, 2]
    # instead of [-3, 2]

    # -----------------------
    # Plot
    # -----------------------
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    axes = axes.flatten()

    for i, col in enumerate(allergen_columns):

        sc.pl.umap(
            adata_umap,
            color=f"{col}_allergen_score",
            # cmap="Reds",
            cmap="RdBu_r",
            ax=axes[i],
            show=False,
            title=col,
            size=point_size,
            frameon=False,
            sort_order=False,
            vmin=vmin,
            vmax=vmax
        )

    plt.tight_layout()

    plt.savefig(
        "Figures/plot1_allergen_expression_2.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()


def plot_allergen_expression_by_label(
    adata,
    label_key="label",
    allergen_column="byMajority",
    point_size=60,
    random_state=0,
    use_zscore=True,
    # cmap="Reds",
    cmap="RdBu_r",
    percentile_clip=(1, 99)
):

    adata = adata.copy()

    # -----------------------
    # Expression matrix
    # -----------------------
    X = adata.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    # -----------------------
    # Aggregate allergen burden
    # -----------------------
    allergen_mask = (
        adata.var[allergen_column]
        .astype(str)
        .eq("Allergen")
        .values
    )

    cell_score = X[:, allergen_mask].sum(axis=1)

    # Optional visualization stabilization
    if use_zscore:
        cell_score = (
            cell_score - cell_score.mean()
        ) / (cell_score.std() + 1e-9)

    score_name = f"{allergen_column}_allergen_score"

    adata.obs[score_name] = cell_score

    # -----------------------
    # Shared color range
    # -----------------------
    if percentile_clip is not None:
        vmin, vmax = np.percentile(
            cell_score,
            percentile_clip
        )
    else:
        vmin, vmax = cell_score.min(), cell_score.max()

    # -----------------------
    # Labels
    # -----------------------
    labels = (
        adata.obs[label_key].cat.categories
        if hasattr(adata.obs[label_key], "cat")
        else adata.obs[label_key].unique()
    )

    # -----------------------
    # Figure
    # -----------------------
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(8 * len(labels), 6)
    )

    if len(labels) == 1:
        axes = [axes]

    # -----------------------
    # Plot per label
    # -----------------------
    for i, lbl in enumerate(labels):

        adata_sub = adata[
            adata.obs[label_key] == lbl
        ].copy()

        sc.pp.neighbors(adata_sub, use_rep="X")
        sc.tl.umap(adata_sub, random_state=random_state)

        sc.pl.umap(
            adata_sub,
            color=score_name,
            cmap=cmap,
            ax=axes[i],
            show=False,
            title=str(lbl),
            size=point_size,
            frameon=False,
            sort_order=False,
            vmin=vmin,
            vmax=vmax
        )

    plt.savefig("Figures/plot_allergen_expression_by_label_2.png", dpi=1500, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_main_volcano(adata, highlight_allergens=False, alpha=0.05, logfc_thresh=0.0):
    result = adata.uns['rank_proteins_groups']

    names = result['names']['Cancer']
    logfc = result['logfoldchanges']['Cancer']
    pvals_adj = result['pvals_adj']['Cancer']

    logfc = np.array(logfc)
    pvals_adj = np.array(pvals_adj)
    neg_log10_p = -np.log10(pvals_adj)

    sig_mask = pvals_adj < alpha

    plt.figure(figsize=(9, 6))

    plt.scatter(logfc, neg_log10_p, color='lightgrey', alpha=0.6)

    plt.scatter(
        logfc[sig_mask],
        neg_log10_p[sig_mask],
        color='red',
        alpha=0.8,
        label='Significant (FDR < 0.05)'
    )

    if highlight_allergens and 'byMajority' in adata.var.columns:
        allergen_mask = adata.var['byMajority'] == 'Allergen'
        allergen_names = adata.var_names[allergen_mask]

        mask = np.isin(names, allergen_names)

        plt.scatter(
            logfc[mask],
            neg_log10_p[mask],
            color='blue',
            alpha=0.9,
            label='Allergens'
        )

    # for i in np.argsort(pvals_adj)[:13]:
    #     plt.text(logfc[i], neg_log10_p[i], names[i], fontsize=7)

    plt.axvline(x=0, color='black', linewidth=1)
    plt.xlabel("Log2 fold change")
    plt.ylabel("-log10 (adjusted p-value)")
    plt.title("Differential protein abundance (Cancer vs Healthy)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_proteins_heatmap(
    adata,
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=100,
    figsize=(12, 10),
    cmap="RdBu_r"
):
    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    sig = (padj < alpha) & (np.abs(logfc) > logfc_thresh)
    proteins = names[sig]

    print(f"Significant proteins: {len(proteins)}")

    if top_n is not None:
        score = -np.log10(padj[sig]) * np.abs(logfc[sig])
        proteins = proteins[np.argsort(score)[-top_n:]]

    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = StandardScaler().fit_transform(X).T

    df = pd.DataFrame(X, index=proteins, columns=adata.obs_names)

    # =========================================================
    # ORDER SAMPLES
    # =========================================================
    groups = adata.obs["label"]

    order = np.argsort(groups.values)
    df = df.iloc[:, order]
    groups = groups.iloc[order]

    # =========================================================
    # TWO SEPARATE TOP BARS (STACKED STYLE)
    # =========================================================
    cancer_bar = groups.map({
        "Cancer": "#ff7f0e",
        "Healthy": "white"
    })

    healthy_bar = groups.map({
        "Cancer": "white",
        "Healthy": "#1f77b4"
    })

    col_colors = pd.DataFrame({
        "Cancer": cancer_bar,
        "Healthy": healthy_bar
    })

    # =========================================================
    # CLUSTERING
    # =========================================================
    row_linkage = linkage(df.values, method="ward")

    vlim = np.max(np.abs(df.values))

    g = sns.clustermap(
        df,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        figsize=figsize,
        xticklabels=False,
        yticklabels=True,
        col_colors=col_colors,
        dendrogram_ratio=(0.15, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.18)
    )

    # =========================================================
    # CLEAN TOP ARTIFACTS
    # =========================================================
    g.ax_col_dendrogram.set_visible(False)

    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.set_ylabel("")
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)

    # =========================================================
    # SEPARATION LINE (optional but recommended)
    # =========================================================
    order_groups = groups.values
    boundary = np.where(order_groups[:-1] != order_groups[1:])[0]

    if len(boundary) > 0:
        for b in boundary:
            g.ax_heatmap.axvline(b + 0.5, color="white", linewidth=3)

    plt.suptitle("Differential Protein Heatmap", y=1.02)
    plt.show()

    return g