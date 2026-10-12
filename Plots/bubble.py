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


def plot_allergen_bubble_impact(
    data,
    figsize=(12, 12),
    title=None
):

    df = pd.DataFrame(data)

    # =========================================================
    # 1. CLEAN + CLIP FIRST (IMPORTANT FIX)
    # =========================================================
    df["logFC"] = df["logFC"].clip(-0.07, 0.07)
    df = df.sort_values(by="logFC", ascending=True).reset_index(drop=True)

    x = df["logFC"].values
    y = np.arange(len(df))

    # =========================================================
    # 2. NORMALIZE -log10FDR → DISCRETE SCALE (1–7)
    # =========================================================
    fdr = df["-log10FDR"].values

    fdr_norm = (fdr - fdr.min()) / (fdr.max() - fdr.min() + 1e-9)
    fdr_scaled = np.round(1 + fdr_norm * 6).astype(int)  # 1–7

    # bubble size mapping (plot)
    size_map = {i: 40 + (i - 1) * 60 for i in range(1, 8)}
    size_scaled = np.array([size_map[v] for v in fdr_scaled])

    # =========================================================
    # 3. COLORS
    # =========================================================
    COLOR_UP = "#EE0000FF"
    COLOR_DOWN = "#3B4992FF"

    colors = np.where(x < 0, COLOR_DOWN, COLOR_UP)

    # =========================================================
    # 4. PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x, y,
        s=size_scaled,
        c=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlim(-0.08, 0.08)
    ax.set_xticks([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    # ax.set_xticklabels(["-0.06", "-0.04", "-0.02", "0.00", "0.02", "+0.05", "+0.07"], fontsize=11)

    # ax.set_yticks(y)
    # ax.set_yticklabels(df["feature"], fontsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.tick_params(axis='y', labelsize=8)

    # axis labels (IMPORTANT UPGRADE)
    # ax.set_xlabel(r"$\log_{2}(\mathrm{fold\ change})$", fontsize=12)
    ax.set_xlabel("Log2FC", fontsize=12)
    ax.set_ylabel("Allergens", fontsize=12)

    # tick styling (makes figure much more publication-grade)
    ax.tick_params(axis='x', labelsize=11, width=1)
    # ax.tick_params(axis='y', labelsize=11, width=1)

    if title is None:
        title = "Top Regulated Allergens"
    ax.set_title(title, fontsize=14)

    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # =========================================================
    # 5. SINGLE UNIFIED LEGEND (RIGHT CENTER, NO COLLISION)
    # =========================================================
    handles = []
    labels = []

    # -------------------------
    # Abundance section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append("Abundance")

    handles += [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_UP, markeredgecolor="black",
               markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_DOWN, markeredgecolor="black",
               markersize=10),
    ]
    labels += ["Upregulated", "Downregulated"]

    handles.append(Line2D([], [], linestyle='none'))
    labels.append(" ")

    # -------------------------
    # Significance section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append(r"$-\log_{10}(p_{\mathrm{adj}})$")

    for v in range(1, 8):
        handles.append(
            Line2D(
                [0], [0],
                marker='o',
                color='black',
                linestyle='none',
                alpha=0.3,
                markersize=np.sqrt(size_map[v])
            )
        )
        labels.append(str(v))

    # for v in range(1, 8):
    #     handles.append(
    #         Line2D([0], [0], marker='o', color='black', alpha=0.3,
    #                linestyle='none',
    #                markersize=np.sqrt(40 + v * 80) / 2)
    #     )
    #     labels.append(str(v))

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        prop={'size': 10},
        labelspacing=1.2,
        borderpad=0.8,
        handletextpad=0.8
    )

    # styling
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_facecolor("white")

    # bold section headers
    section_titles = {"Abundance", r"$-\log_{10}(p_{\mathrm{adj}})$"}

    for text in leg.get_texts():
        if text.get_text().strip() in section_titles:
            if not text.get_text().strip() == r"$-\log_{10}(p_{\mathrm{adj}})$":
                text.set_fontsize(13)
                text.set_fontweight("bold")
            else:
                text.set_fontsize(11.5)
        # if text.get_text().strip() in section_titles:
        #     text.set_fontweight("bold")
        #     text.set_fontsize(14)

    # layout (space for legend)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(
        "Figures/plot_top_regulated_allergens_bubble_plot4.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()

    return df


data = [
    {"feature": "A0A239YF41_9FIRM", "logFC": -0.02955, "-log10FDR": 5.754},
    {"feature": "E6KWI6_9PAST", "logFC": -0.02811, "-log10FDR": 4.442},
    {"feature": "D6GSY8_FILAD", "logFC": -0.06503, "-log10FDR": 4.252},
    {"feature": "X8ISL9_9STRE", "logFC": -0.01777, "-log10FDR": 3.637},
    {"feature": "A0A2K9HAQ2_9BACT", "logFC": -0.01731, "-log10FDR": 3.518},
    {"feature": "A0A6L5DSP1_9BACT", "logFC": -0.02142, "-log10FDR": 3.199},
    {"feature": "UPI000469B5B7", "logFC": -0.01600, "-log10FDR": 3.047},
    {"feature": "A0A3N2MZC8_9BACT", "logFC": -0.03159, "-log10FDR": 2.915},
    {"feature": "A0A379EY57_9BACT", "logFC": -0.02975, "-log10FDR": 2.915},
    {"feature": "L1MCS5_9BACT", "logFC": -0.01899, "-log10FDR": 2.834},

    {"feature": "D1BP93_VEIPT", "logFC": 0.03986, "-log10FDR": 5.059},
    {"feature": "F3B2V5_9FIRM", "logFC": 0.03836, "-log10FDR": 2.862},
    {"feature": "A0A2W5KWT7_9CORY", "logFC": 0.02282, "-log10FDR": 2.726},
    {"feature": "A0A2S7ZLE7_9FIRM", "logFC": 0.03866, "-log10FDR": 2.666},
    {"feature": "W1VMM1_STRPA", "logFC": 0.01489, "-log10FDR": 2.272},
    {"feature": "D2ZSM3_NEIMU", "logFC": 0.02536, "-log10FDR": 2.083},
    {"feature": "A0A2T4TA75_9FIRM", "logFC": 0.02578, "-log10FDR": 2.014},
    {"feature": "H1HWB9_9FIRM", "logFC": 0.02092, "-log10FDR": 1.968},
    {"feature": "A0A1E9MPU5_9NEIS", "logFC": 0.03474, "-log10FDR": 1.961},
    {"feature": "K0XE54_9FIRM", "logFC": 0.02809, "-log10FDR": 1.948},
]

plot_allergen_bubble_impact(data)