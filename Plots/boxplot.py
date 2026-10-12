import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from os.path import exists
from os.path import join
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import anndata
import json
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.stats import gaussian_kde
from scipy.stats import fisher_exact
import matplotlib.colors as colors
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


adata_with_allergens_path = "../H5AD/adata_with_allergens.h5ad"
adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)

plt.rcParams['font.family'] = 'Times New Roman'


def plot_top_upregulated_allergen_boxplots_final(
    adata,
    group_key="label",
    group_pos="Cancer",
    group_neg="Healthy",
    allergen_key="byMajority",
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=10
):

    # COLOR_CANCER = "#E64B35"
    # COLOR_HEALTHY = "#4DBBD5"
    COLOR_HEALTHY = "#1f77b4"
    COLOR_CANCER = "#ff7f0e"

    flierprops = dict(
        marker='o',
        markerfacecolor='none',
        markeredgecolor='black',
        markersize=5,
        linestyle='none'
    )

    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    is_allergen = adata.var.loc[names, allergen_key].values == "Allergen"

    sel = (padj < alpha) & (logfc > logfc_thresh) & is_allergen

    idx = np.where(sel)[0]
    # idx = idx[np.argsort(logfc[idx])[::-1]]
    idx = idx[np.argsort(padj[idx])]
    top_idx = idx[:top_n]

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    groups = adata.obs[group_key].values

    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 8),
        gridspec_kw={"wspace": 0.6, "hspace": 0.30}
    )

    axes = axes.flatten()

    for k, i in enumerate(top_idx):

        ax = axes[k]

        gene = names[i]
        j = adata.var_names.get_loc(gene)

        cancer_vals = X[groups == group_pos, j]
        healthy_vals = X[groups == group_neg, j]

        data = [healthy_vals, cancer_vals]

        bp = ax.boxplot(
            data,
            labels=[group_neg, group_pos],
            patch_artist=True,
            showfliers=True,
            flierprops=flierprops,
            widths=0.65,
            medianprops=dict(color='black', linewidth=1, linestyle='--')
        )

        bp["boxes"][0].set_facecolor(COLOR_HEALTHY)
        bp["boxes"][1].set_facecolor(COLOR_CANCER)

        for box in bp["boxes"]:
            box.set_alpha(0.75)

        # =========================================================
        # IQR FILTERING FOR SCATTER POINTS (KEY CHANGE)
        # =========================================================
        def iqr_mask(vals):
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return (vals >= low) & (vals <= high)

        mask_h = iqr_mask(healthy_vals)
        mask_c = iqr_mask(cancer_vals)

        # scatter ONLY in-range values
        ax.scatter(
            np.random.normal(1, 0.04, mask_h.sum()),
            healthy_vals[mask_h],
            color=COLOR_HEALTHY,
            alpha=0.5,
            s=10
        )

        ax.scatter(
            np.random.normal(2, 0.04, mask_c.sum()),
            cancer_vals[mask_c],
            color=COLOR_CANCER,
            alpha=0.5,
            s=10
        )

        # =========================================================
        # STARS + P-VALUE
        # =========================================================
        p = padj[i]

        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"

        p_label = f"{p:.3e}"

        # =========================================================
        # ROBUST Y LIMIT
        # =========================================================
        y_all = np.concatenate([healthy_vals, cancer_vals])

        q1 = np.percentile(y_all, 25)
        q3 = np.percentile(y_all, 75)
        iqr = q3 - q1

        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        data_min = min(y_all.min(), lower_whisker)
        data_max = max(y_all.max(), upper_whisker)

        y_range = (data_max - data_min + 1e-9)

        pad_bottom = max(0.10 * y_range, 0.05 * np.abs(data_min))
        pad_top = max(0.40 * y_range, 0.10 * np.abs(data_max))

        y_bottom = data_min - pad_bottom
        y_top = data_max + pad_top

        ax.set_ylim(y_bottom, y_top)

        ax.text(
            1.5,
            y_top - 0.05 * (y_top - y_bottom),
            f"{star}\n(Padj={p_label})",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

        ax.set_title(f"{k+1}. {gene}", fontsize=10)

    for k in range(len(top_idx), len(axes)):
        axes[k].axis("off")

    legend_elements = [
        Line2D([], [], color='none', label="Groups:"),
        Patch(facecolor=COLOR_HEALTHY, label=group_neg),
        Patch(facecolor=COLOR_CANCER, label=group_pos)
    ]

    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12
    )

    for text in leg.get_texts():
        if text.get_text() == "Groups:":
            text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = '/Users/pedram/Desktop/Project/Figures/plot_top_10_upregulated_allergens_in_cancer_boxplots_fixed.png'
    plt.savefig(save_path, dpi=1500, bbox_inches="tight")
    plt.show()


def plot_top_downregulated_allergen_boxplots_final(
    adata,
    group_key="label",
    group_pos="Cancer",
    group_neg="Healthy",
    allergen_key="byMajority",
    alpha=0.05,
    logfc_thresh=0.01,
    top_n=10
):

    COLOR_HEALTHY = "#1f77b4"
    COLOR_CANCER = "#ff7f0e"

    flierprops = dict(
        marker='o',
        markerfacecolor='none',
        markeredgecolor='black',
        markersize=5,
        linestyle='none'
    )

    res = adata.uns["rank_proteins_groups"]

    names = np.array(res["names"]["Cancer"])
    logfc = np.array(res["logfoldchanges"]["Cancer"])
    padj = np.clip(np.array(res["pvals_adj"]["Cancer"]), 1e-300, None)

    is_allergen = adata.var.loc[names, allergen_key].values == "Allergen"

    # =========================================================
    # CHANGE: LEFT SIDE VOLCANO (DOWN in cancer)
    # =========================================================
    sel = (padj < alpha) & (logfc < -logfc_thresh) & is_allergen

    idx = np.where(sel)[0]

    # most negative logFC first (strongest downregulation in cancer)
    idx = idx[np.argsort(padj[idx])]
    # idx = idx[np.argsort(logfc[idx])]
    top_idx = idx[:top_n]

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    groups = adata.obs[group_key].values

    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 8),
        gridspec_kw={"wspace": 0.6, "hspace": 0.30}
    )

    axes = axes.flatten()

    for k, i in enumerate(top_idx):

        ax = axes[k]

        gene = names[i]
        j = adata.var_names.get_loc(gene)

        cancer_vals = X[groups == group_pos, j]
        healthy_vals = X[groups == group_neg, j]

        data = [healthy_vals, cancer_vals]

        bp = ax.boxplot(
            data,
            labels=[group_neg, group_pos],
            patch_artist=True,
            showfliers=True,
            flierprops=flierprops,
            widths=0.65,
            medianprops=dict(color='black', linewidth=1, linestyle='--')
        )

        bp["boxes"][0].set_facecolor(COLOR_HEALTHY)
        bp["boxes"][1].set_facecolor(COLOR_CANCER)

        for box in bp["boxes"]:
            box.set_alpha(0.75)

        # =========================================================
        # IQR FILTERING (UNCHANGED)
        # =========================================================
        def iqr_mask(vals):
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return (vals >= low) & (vals <= high)

        mask_h = iqr_mask(healthy_vals)
        mask_c = iqr_mask(cancer_vals)

        ax.scatter(
            np.random.normal(1, 0.04, mask_h.sum()),
            healthy_vals[mask_h],
            color=COLOR_HEALTHY,
            alpha=0.5,
            s=10
        )

        ax.scatter(
            np.random.normal(2, 0.04, mask_c.sum()),
            cancer_vals[mask_c],
            color=COLOR_CANCER,
            alpha=0.5,
            s=10
        )

        # =========================================================
        # STARS + P-VALUE
        # =========================================================
        p = padj[i]

        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"

        p_label = f"{p:.3e}"

        # =========================================================
        # ROBUST Y LIMIT
        # =========================================================
        y_all = np.concatenate([healthy_vals, cancer_vals])

        q1 = np.percentile(y_all, 25)
        q3 = np.percentile(y_all, 75)
        iqr = q3 - q1

        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        data_min = min(y_all.min(), lower_whisker)
        data_max = max(y_all.max(), upper_whisker)

        y_range = (data_max - data_min + 1e-9)

        pad_bottom = max(0.10 * y_range, 0.05 * np.abs(data_min))
        pad_top = max(0.40 * y_range, 0.10 * np.abs(data_max))

        y_bottom = data_min - pad_bottom
        y_top = data_max + pad_top

        ax.set_ylim(y_bottom, y_top)

        ax.text(
            1.5,
            y_top - 0.05 * (y_top - y_bottom),
            f"{star}\n(Padj={p_label})",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

        ax.set_title(f"{k+1}. {gene}", fontsize=10)

    for k in range(len(top_idx), len(axes)):
        axes[k].axis("off")

    legend_elements = [
        Line2D([], [], color='none', label="Groups:"),
        Patch(facecolor=COLOR_HEALTHY, label=group_neg),
        Patch(facecolor=COLOR_CANCER, label=group_pos)
    ]

    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12
    )

    for text in leg.get_texts():
        if text.get_text() == "Groups:":
            text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = "/Users/pedram/Desktop/Project/Figures/plot_top_10_downregulated_allergens_in_cancer_boxplots_fixed.png"
    plt.savefig(
        save_path,
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()


plot_top_upregulated_allergen_boxplots_final(adata_ranked_proteins)
plot_top_downregulated_allergen_boxplots_final(adata_ranked_proteins)

# print(adata_ranked_proteins.obs["label"])


# [DEBUG] Top Downregulated allergens (left side)
# ======================================================================
# 01 | A0A239YF41_9FIRM | logFC=-0.02955 | -log10FDR=5.754
# 02 | E6KWI6_9PAST | logFC=-0.02811 | -log10FDR=4.442
# 03 | D6GSY8_FILAD | logFC=-0.06503 | -log10FDR=4.252
# 04 | X8ISL9_9STRE | logFC=-0.01777 | -log10FDR=3.637
# 05 | A0A2K9HAQ2_9BACT | logFC=-0.01731 | -log10FDR=3.518
# 06 | A0A6L5DSP1_9BACT | logFC=-0.02142 | -log10FDR=3.199
# 07 | UPI000469B5B7 | logFC=-0.01600 | -log10FDR=3.047
# 08 | A0A3N2MZC8_9BACT | logFC=-0.03159 | -log10FDR=2.915
# 09 | A0A379EY57_9BACT | logFC=-0.02975 | -log10FDR=2.915
# 10 | L1MCS5_9BACT | logFC=-0.01899 | -log10FDR=2.834
#
# [DEBUG] Top Upregulated allergens (right side)
# ======================================================================
# 01 | D1BP93_VEIPT | logFC=0.03986 | -log10FDR=5.059
# 02 | F3B2V5_9FIRM | logFC=0.03836 | -log10FDR=2.862
# 03 | A0A2W5KWT7_9CORY | logFC=0.02282 | -log10FDR=2.726
# 04 | A0A2S7ZLE7_9FIRM | logFC=0.03866 | -log10FDR=2.666
# 05 | W1VMM1_STRPA | logFC=0.01489 | -log10FDR=2.272
# 06 | D2ZSM3_NEIMU | logFC=0.02536 | -log10FDR=2.083
# 07 | A0A2T4TA75_9FIRM | logFC=0.02578 | -log10FDR=2.014
# 08 | H1HWB9_9FIRM | logFC=0.02092 | -log10FDR=1.968
# 09 | A0A1E9MPU5_9NEIS | logFC=0.03474 | -log10FDR=1.961
# 10 | K0XE54_9FIRM | logFC=0.02809 | -log10FDR=1.948
