import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from os.path import exists
from os.path import join
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata
import json
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.stats import gaussian_kde
from scipy.stats import fisher_exact
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage


mpl.rcParams['font.family'] = 'Times New Roman'
adata_with_allergens_path = "../H5AD/adata_with_allergens.h5ad"
adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)
print(adata_ranked_proteins)


def plot_proteins_heatmap(
    adata,
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=80,
    figsize=(12, 10),
    cmap="RdBu_r"
):

    # =========================================================
    # 1. DE RESULTS
    # =========================================================
    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    neglog = -np.log10(padj)

    # =========================================================
    # 2. STRICT FILTER (MATCH BOXPLOT LOGIC)
    # =========================================================
    sel = (padj < alpha) & (logfc > logfc_thresh)

    proteins_idx = np.where(sel)[0]

    print(f"Significant proteins (Cancer-up): {len(proteins_idx)}")

    if len(proteins_idx) == 0:
        raise ValueError("No proteins pass DE filters.")

    # =========================================================
    # 3. VOLCANO-BASED RANKING (IMPORTANT FIX)
    # =========================================================
    proteins_idx = proteins_idx[np.argsort(neglog[proteins_idx])[::-1]]

    if top_n is not None:
        proteins_idx = proteins_idx[:top_n]

    proteins = [names[i] for i in proteins_idx if names[i] in adata.var_names]

    # =========================================================
    # 4. EXPRESSION MATRIX
    # =========================================================
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # row-wise scaling (protein-wise z-score)
    X = StandardScaler().fit_transform(X).T

    df = pd.DataFrame(X, index=proteins, columns=adata.obs_names)

    # =========================================================
    # 5. ORDER SAMPLES
    # =========================================================
    groups = adata.obs["label"]

    order = np.argsort(groups.values)
    df = df.iloc[:, order]
    groups = groups.iloc[order]

    # =========================================================
    # 6. CLEAN SINGLE COLOR BAR (FIXED)
    # =========================================================
    palette = {
        "Cancer": "#ff7f0e",
        "Healthy": "#1f77b4"
    }

    col_colors = groups.map(palette)

    # =========================================================
    # 7. CLUSTERING
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
    # 8. CLEAN UI
    # =========================================================
    g.ax_col_dendrogram.set_visible(False)

    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.set_ylabel("")
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)

    # =========================================================
    # 9. GROUP SEPARATION LINE
    # =========================================================
    order_groups = groups.values
    boundary = np.where(order_groups[:-1] != order_groups[1:])[0]

    for b in boundary:
        g.ax_heatmap.axvline(b + 0.5, color="white", linewidth=2)

    plt.suptitle("Top Differential Cancer-Upregulated Proteins", y=1.02)
    plt.show()

    return g


def plot_proteins_heatmap2(
    adata,
    allergen_key="byMajority",
    allergen_value="Allergen",
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=10,
    figsize=(12, 10),
    cmap="RdBu_r"
):

    # =========================================================
    # 1. DE RESULTS
    # =========================================================
    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    neglog = -np.log10(padj)

    # =========================================================
    # 2. ALLERGEN FILTER (CRITICAL FIX)
    # =========================================================
    if allergen_key not in adata.var.columns:
        raise ValueError(f"{allergen_key} not found in adata.var")

    is_allergen = (
        adata.var.loc[names, allergen_key]
        .astype(str)
        .values == allergen_value
    )

    # =========================================================
    # 3. STRICT DE FILTER (Cancer-UP + Allergen)
    # =========================================================
    sel = (padj < alpha) & (logfc > logfc_thresh) & is_allergen

    idx = np.where(sel)[0]

    print(f"Significant allergen proteins (Cancer-up): {len(idx)}")

    if len(idx) == 0:
        raise ValueError("No allergen proteins pass DE filters.")

    # =========================================================
    # 4. VOLCANO-BASED RANKING (CONSISTENT WITH YOUR LOGIC)
    # =========================================================
    idx = idx[np.argsort(neglog[idx])[::-1]]

    if top_n is not None:
        idx = idx[:top_n]

    proteins = [names[i] for i in idx if names[i] in adata.var_names]

    # =========================================================
    # 5. EXPRESSION MATRIX
    # =========================================================
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = StandardScaler().fit_transform(X).T

    df = pd.DataFrame(X, index=proteins, columns=adata.obs_names)

    # =========================================================
    # 6. SAMPLE ORDERING
    # =========================================================
    groups = adata.obs["label"]

    order = np.argsort(groups.values)
    df = df.iloc[:, order]
    groups = groups.iloc[order]

    # =========================================================
    # 7. COLOR BAR (FIXED SINGLE BAR)
    # =========================================================
    palette = {
        "Cancer": "#ff7f0e",
        "Healthy": "#1f77b4"
    }

    col_colors = groups.map(palette)

    # =========================================================
    # 8. CLUSTERING
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
    # 9. CLEAN UI
    # =========================================================
    g.ax_col_dendrogram.set_visible(False)

    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.set_ylabel("")
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)

    # =========================================================
    # 10. GROUP SEPARATION LINE
    # =========================================================
    order_groups = groups.values
    boundary = np.where(order_groups[:-1] != order_groups[1:])[0]

    for b in boundary:
        g.ax_heatmap.axvline(b + 0.5, color="white", linewidth=2)

    plt.suptitle("Cancer-Upregulated Allergen Proteins", y=1.02)
    plt.show()

    return g


def plot_proteins_heatmap3_up(
    adata,
    allergen_key="byMajority",
    allergen_value="Allergen",
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=100,
    figsize=(12, 10),
    cmap="RdBu_r"
):

    # =========================================================
    # 1. DE RESULTS
    # =========================================================
    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    neglog = -np.log10(padj)

    # =========================================================
    # 2. ALLERGEN FILTER
    # =========================================================
    if allergen_key not in adata.var.columns:
        raise ValueError(f"{allergen_key} not found in adata.var")

    is_allergen = (
        adata.var.loc[names, allergen_key]
        .astype(str)
        .values == allergen_value
    )

    # =========================================================
    # 3. STRICT DE FILTER
    # =========================================================
    sel = (padj < alpha) & (logfc > logfc_thresh) & is_allergen

    idx = np.where(sel)[0]

    print(f"Significant allergen proteins (Cancer-up): {len(idx)}")

    if len(idx) == 0:
        raise ValueError("No allergen proteins pass DE filters.")

    # =========================================================
    # 4. VOLCANO RANKING
    # =========================================================
    idx = idx[np.argsort(neglog[idx])[::-1]]

    if top_n is not None:
        idx = idx[:top_n]

    proteins = [names[i] for i in idx if names[i] in adata.var_names]

    # =========================================================
    # 5. EXPRESSION MATRIX
    # =========================================================
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = StandardScaler().fit_transform(X).T

    df = pd.DataFrame(X, index=proteins, columns=adata.obs_names)

    # =========================================================
    # 6. SAMPLE ORDERING
    # =========================================================
    groups = adata.obs["label"]

    order = np.argsort(groups.values)
    # order = np.argsort(groups.values == "Healthy")[::-1]
    df = df.iloc[:, order]
    groups = groups.iloc[order]

    # =========================================================
    # 7. COLOR BAR
    # =========================================================
    palette = {
        "Cancer": "#ff7f0e",
        "Healthy": "#1f77b4"
    }

    col_colors = groups.map(palette)

    # =========================================================
    # 8. CLUSTERING
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

    g.cax.set_title("Z-score", fontsize=12, pad=10)

    # =========================================================
    # 9. CLEAN UI
    # =========================================================
    g.ax_col_dendrogram.set_visible(False)

    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.set_ylabel("")
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)

    # =========================================================
    # 10. GROUP SEPARATION LINE
    # =========================================================
    order_groups = groups.values
    boundary = np.where(order_groups[:-1] != order_groups[1:])[0]

    for b in boundary:
        g.ax_heatmap.axvline(b + 0.5, color="white", linewidth=2)

    # =========================================================
    # 11. LABELS ABOVE COLOR BAR (ROBUST FIX)
    # =========================================================
    ax = g.ax_col_colors

    ax.text(0.25, 1.4, "Healthy",
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=palette["Healthy"],
            fontsize=15, fontweight="bold")

    ax.text(0.75, 1.4, "Cancer",
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=palette["Cancer"],
            fontsize=15, fontweight="bold")

    # =========================================================
    # TITLE
    # =========================================================
    # plt.suptitle("Cancer-Upregulated Allergen Proteins", y=1.02)
    plt.savefig(
        "Figures/plot_heatmap_top_regulated_allergens_in_cancer.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()

    return g


def plot_proteins_heatmap3_down(
    adata,
    allergen_key="byMajority",
    allergen_value="Allergen",
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=100,
    figsize=(12, 10),
    cmap="RdBu_r"
):

    # =========================================================
    # 1. DE RESULTS
    # =========================================================
    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    neglog = -np.log10(padj)

    # =========================================================
    # 2. ALLERGEN FILTER
    # =========================================================
    if allergen_key not in adata.var.columns:
        raise ValueError(f"{allergen_key} not found in adata.var")

    is_allergen = (
        adata.var.loc[names, allergen_key]
        .astype(str)
        .values == allergen_value
    )

    # =========================================================
    # 3. STRICT DE FILTER (Cancer-DOWN + Allergen)
    # =========================================================
    sel = (padj < alpha) & (logfc < -logfc_thresh) & is_allergen

    idx = np.where(sel)[0]

    print(f"Significant allergen proteins (Cancer-down): {len(idx)}")

    if len(idx) == 0:
        raise ValueError("No allergen proteins pass DE filters.")

    # =========================================================
    # 4. VOLCANO RANKING (strongest DOWN first)
    # =========================================================
    idx = idx[np.argsort(neglog[idx])[::-1]]

    if top_n is not None:
        idx = idx[:top_n]

    proteins = [names[i] for i in idx if names[i] in adata.var_names]

    # =========================================================
    # 5. EXPRESSION MATRIX
    # =========================================================
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = StandardScaler().fit_transform(X).T

    df = pd.DataFrame(X, index=proteins, columns=adata.obs_names)

    # =========================================================
    # 6. SAMPLE ORDERING
    # =========================================================
    groups = adata.obs["label"]

    order = np.argsort(groups.values)
    df = df.iloc[:, order]
    groups = groups.iloc[order]

    # =========================================================
    # 7. COLOR BAR
    # =========================================================
    palette = {
        "Cancer": "#ff7f0e",
        "Healthy": "#1f77b4"
    }

    col_colors = groups.map(palette)

    # =========================================================
    # 8. CLUSTERING
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

    g.cax.set_position([0.02, 0.85, 0.03, 0.12])
    g.cax.set_title("Z-score", fontsize=12, pad=10)

    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=4 # for 100 proteins
    )

    # =========================================================
    # 9. CLEAN UI
    # =========================================================
    g.ax_col_dendrogram.set_visible(False)

    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.set_ylabel("")
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)

    # =========================================================
    # 10. GROUP SEPARATION LINE
    # =========================================================
    order_groups = groups.values
    boundary = np.where(order_groups[:-1] != order_groups[1:])[0]

    for b in boundary:
        g.ax_heatmap.axvline(b + 0.5, color="white", linewidth=2)

    # =========================================================
    # 11. LABELS ABOVE COLOR BAR
    # =========================================================
    ax = g.ax_col_colors

    ax.text(0.25, 1.4, "Healthy",
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=palette["Healthy"],
            fontsize=15, fontweight="bold")

    ax.text(0.75, 1.4, "Cancer",
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=palette["Cancer"],
            fontsize=15, fontweight="bold")

    # =========================================================
    # TITLE
    # =========================================================
    # plt.suptitle("Cancer-Downregulated Allergen Proteins", y=1.02)
    plt.savefig(
        "Figures/plot_heatmap_top_downregulated_allergens_in_cancer.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()

    return g


# plot_proteins_heatmap3_up(adata_ranked_proteins, top_n=100)
plot_proteins_heatmap3_down(adata_ranked_proteins, top_n=100)
