import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from adjustText import adjust_text
from os.path import exists
from os.path import join
import pandas as pd
import anndata
import json

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


def plot_main_volcano7(adata, alpha=0.05, logfc_thresh=0.01, figsize=(12, 8)):
    # mpl.rcParams.update({
    #     'font.family': 'Times New Roman',
    #     'axes.titleweight': 'regular',
    #     'axes.labelsize': 13,
    #     'axes.titlesize': 15,
    #     'xtick.labelsize': 12,
    #     'ytick.labelsize': 12,
    #     'legend.fontsize': 11,
    # })

    COLOR_UP = "#EE0000FF"
    COLOR_DOWN = "#3B4992FF"
    COLOR_NEUTRAL = "#808180FF"

    # COLOR_DOWN = "#1f77b4"
    # COLOR_UP = "#ff7f0e"
    # COLOR_NEUTRAL = "#808180FF"

    # =========================================================
    # DATA
    # =========================================================
    result = adata.uns['rank_proteins_groups']
    names = np.array(result['names']['Cancer'])
    logfc = np.array(result['logfoldchanges']['Cancer'])
    pvals_adj = np.clip(np.array(result['pvals_adj']['Cancer']), 1e-300, None)

    neg_log10_p = -np.log10(pvals_adj)

    sig_mask = neg_log10_p > -np.log10(alpha)
    up_mask = (logfc > logfc_thresh) & sig_mask
    down_mask = (logfc < -logfc_thresh) & sig_mask

    size = 20 + (neg_log10_p * 18)
    is_allergen = adata.var.loc[names, 'byMajority'].values == 'Allergen'

    fig, ax = plt.subplots(figsize=figsize)

    # =========================================================
    # SCATTER
    # =========================================================
    def scatter(mask, marker):
        ax.scatter(
            logfc[mask],
            neg_log10_p[mask],
            c=np.where(
                up_mask[mask], COLOR_UP,
                np.where(down_mask[mask], COLOR_DOWN, COLOR_NEUTRAL)
            ),
            s=size[mask],
            alpha=0.75,
            marker=marker,
            edgecolor='none'
        )

    scatter(~is_allergen, "o")
    scatter(is_allergen, "*")

    # =========================================================
    # THRESHOLDS
    # =========================================================
    ax.axhline(-np.log10(alpha), color='black', linestyle='--', linewidth=1)
    ax.text(
        ax.get_xlim()[1],
        -np.log10(alpha),
        f"  FDR = {alpha}",
        va='bottom',
        ha='left',
        fontsize=10
    )
    ax.axvline(logfc_thresh, color='black', linestyle='--', linewidth=1)
    ax.axvline(-logfc_thresh, color='black', linestyle='--', linewidth=1)
    ax.text(
        -logfc_thresh,
        ax.get_ylim()[1],
        f"log2FC = -{logfc_thresh}  ",
        rotation=90,
        va='top',
        ha='right',
        fontsize=10
    )
    ax.axvline(0, color='black', linewidth=0.8)

    ax.set_xlim(-0.07, 0.07)

    ax.set_xlabel(r"$\log_{2}(\mathrm{fold\ change})$", fontsize=13)
    ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{adj}})$", fontsize=13)
    ax.set_title("Differential protein abundance (cancer vs healthy)")
    fig.subplots_adjust(right=0.72)

    # =========================================================
    # SINGLE UNIFIED LEGEND (NO OVERLAP GUARANTEED)
    # =========================================================
    COLOR_UP = "#EE0000FF"
    COLOR_DOWN = "#3B4992FF"
    COLOR_NEUTRAL = "#808180FF"

    handles = []
    labels = []

    # -------------------------
    # Proteins section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append("Abundance")

    # markersize = 10
    handles += [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_UP, markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_DOWN, markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_NEUTRAL, markersize=10)
    ]
    labels += ["Increased", "Decreased", "Not significant"]
    handles.append(Line2D([], [], linestyle='none'))
    labels.append(" ")
    # handles.append(Line2D([], [], linestyle='none'))
    # labels.append(" ")

    # -------------------------
    # Allergens section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append("Allergens")

    # markersize = 13
    handles += [
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_UP, markersize=15),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_DOWN, markersize=15),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_NEUTRAL, markersize=15),
    ]
    labels += ["Increased", "Decreased", "Not significant"]
    handles.append(Line2D([], [], linestyle='none'))
    labels.append(" ")

    # -------------------------
    # Significance section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    # labels.append(r"$-\log_{10}(p_{\mathrm{adj}})$")
    labels.append(r"$\boldsymbol{-\log_{10}(p_{\mathrm{adj}})}$")

    # np.sqrt(20 + v * 18) / 2)
    for v in range(1, 8):
        handles.append(
            Line2D([0], [0], marker='o', color='black', alpha=0.3,
                   linestyle='none',
                   markersize=np.sqrt(40 + v * 80) / 2)
        )
        labels.append(str(v))

    # -------------------------
    # CREATE SINGLE LEGEND
    # -------------------------
    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        # frameon=False,
        prop={'size': 10},
        labelspacing=0.8,
        borderpad=0.8
    )

    # style frame
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_facecolor("white")

    # make section headers bold
    # section_titles = {"Proteins", "Significance", "Allergens"}
    section_titles = {"Abundance", r"$-\log_{10}(p_{\mathrm{adj}})$", "Allergens"}
    for text in leg.get_texts():
        if text.get_text().strip() in section_titles:
            if not text.get_text().strip() == r"$-\log_{10}(p_{\mathrm{adj}})$":
                text.set_fontsize(13)
                text.set_fontweight("bold")
            else:
                text.set_fontsize(11.5)
            # text.set_fontweight("bold")

    # =========================================================
    # SHOW
    # =========================================================
    plt.tight_layout()
    plt.savefig("Figures/volcano_plot.png", dpi=1500, bbox_inches="tight")
    plt.show()


def plot_main_volcano_final(
    adata,
    alpha=0.05,
    logfc_thresh=0.01,
    figsize=(12, 8),
    top_n_labels=10
):

    COLOR_UP = "#EE0000FF"
    COLOR_DOWN = "#3B4992FF"
    COLOR_NEUTRAL = "#808180FF"

    # =========================================================
    # DATA
    # =========================================================
    result = adata.uns['rank_proteins_groups']
    names = np.array(result['names']['Cancer'])
    logfc = np.array(result['logfoldchanges']['Cancer'])

    pvals_adj = np.clip(
        np.array(result['pvals_adj']['Cancer']),
        1e-300,
        None
    )

    neg_log10_p = -np.log10(pvals_adj)

    sig_mask = neg_log10_p > -np.log10(alpha)
    up_mask = (logfc > logfc_thresh) & sig_mask
    down_mask = (logfc < -logfc_thresh) & sig_mask

    size = 20 + (neg_log10_p * 18)

    is_allergen = adata.var.loc[names, 'byMajority'].values == 'Allergen'

    # =========================================================
    # FIGURE
    # =========================================================
    fig, ax = plt.subplots(figsize=figsize)

    # =========================================================
    # SCATTER
    # =========================================================
    def scatter(mask, marker):
        ax.scatter(
            logfc[mask],
            neg_log10_p[mask],
            c=np.where(
                up_mask[mask], COLOR_UP,
                np.where(down_mask[mask], COLOR_DOWN, COLOR_NEUTRAL)
            ),
            s=size[mask],
            alpha=0.75,
            marker=marker,
            edgecolor='none'
        )

    scatter(~is_allergen, "o")
    scatter(is_allergen, "*")

    # =========================================================
    # LEFT SIDE LABELS (DOWNREGULATED)
    # =========================================================
    left_mask = (
        is_allergen &
        (logfc < -logfc_thresh) &
        sig_mask
    )

    left_idx = np.where(left_mask)[0]
    left_idx = left_idx[np.argsort(neg_log10_p[left_idx])[::-1]]
    left_top = left_idx[:top_n_labels]

    texts = []

    print("\n[DEBUG] Top DOWNregulated allergens (left side)")
    print("=" * 70)

    for r, i in enumerate(left_top, 1):
        print(f"{r:02d} | {names[i]} | logFC={logfc[i]:.5f} | -log10FDR={neg_log10_p[i]:.3f}")

        texts.append(
            ax.text(
                logfc[i],
                neg_log10_p[i],
                names[i],
                fontsize=9,
                weight='bold'
            )
        )

    # =========================================================
    # RIGHT SIDE LABELS (UPREGULATED)
    # =========================================================
    right_mask = (
        is_allergen &
        (logfc > logfc_thresh) &
        sig_mask
    )

    right_idx = np.where(right_mask)[0]
    right_idx = right_idx[np.argsort(neg_log10_p[right_idx])[::-1]]
    right_top = right_idx[:top_n_labels]

    print("\n[DEBUG] Top UPregulated allergens (right side)")
    print("=" * 70)

    for r, i in enumerate(right_top, 1):
        print(f"{r:02d} | {names[i]} | logFC={logfc[i]:.5f} | -log10FDR={neg_log10_p[i]:.3f}")

        texts.append(
            ax.text(
                logfc[i],
                neg_log10_p[i],
                names[i],
                fontsize=9,
                weight='bold'
            )
        )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', lw=0.5, color='black')
    )

    # =========================================================
    # THRESHOLDS
    # =========================================================
    ax.axhline(-np.log10(alpha), color='black', linestyle='--', linewidth=1)

    ax.text(
        ax.get_xlim()[1],
        -np.log10(alpha),
        f"  FDR = {alpha}",
        va='bottom',
        ha='left',
        fontsize=10
    )

    ax.axvline(logfc_thresh, color='black', linestyle='--', linewidth=1)
    ax.axvline(-logfc_thresh, color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linewidth=0.8)

    # =========================================================
    # FIXED X LIMITS (REQUESTED)
    # =========================================================
    ax.set_xlim(-0.07, 0.07)

    # =========================================================
    # LABELS
    # =========================================================
    ax.set_xlabel(r"$\log_{2}(\mathrm{fold\ change})$", fontsize=13)
    ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{adj}})$", fontsize=13)
    ax.set_title("Differential protein abundance (cancer vs healthy)")

    fig.subplots_adjust(right=0.72)


    # =========================================================
    # SINGLE UNIFIED LEGEND (NO OVERLAP GUARANTEED)
    # =========================================================
    COLOR_UP = "#EE0000FF"
    COLOR_DOWN = "#3B4992FF"
    COLOR_NEUTRAL = "#808180FF"

    handles = []
    labels = []

    # -------------------------
    # Proteins section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append("Abundance")

    # markersize = 10
    handles += [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_UP, markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_DOWN, markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_NEUTRAL, markersize=10)
    ]
    labels += ["Increased", "Decreased", "Not significant"]
    handles.append(Line2D([], [], linestyle='none'))
    labels.append(" ")
    # handles.append(Line2D([], [], linestyle='none'))
    # labels.append(" ")

    # -------------------------
    # Allergens section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    labels.append("Allergens")

    # markersize = 13
    handles += [
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_UP, markersize=15),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_DOWN, markersize=15),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=COLOR_NEUTRAL, markersize=15),
    ]
    labels += ["Increased", "Decreased", "Not significant"]
    handles.append(Line2D([], [], linestyle='none'))
    labels.append(" ")

    # -------------------------
    # Significance section
    # -------------------------
    handles.append(Line2D([], [], linestyle='none'))
    # labels.append(r"$-\log_{10}(p_{\mathrm{adj}})$")
    labels.append(r"$\boldsymbol{-\log_{10}(p_{\mathrm{adj}})}$")

    # np.sqrt(20 + v * 18) / 2)
    for v in range(1, 8):
        handles.append(
            Line2D([0], [0], marker='o', color='black', alpha=0.3,
                   linestyle='none',
                   markersize=np.sqrt(40 + v * 80) / 2)
        )
        labels.append(str(v))

    # -------------------------
    # CREATE SINGLE LEGEND
    # -------------------------
    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        # frameon=False,
        prop={'size': 10},
        labelspacing=0.8,
        borderpad=0.8
    )

    # style frame
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_facecolor("white")

    # make section headers bold
    # section_titles = {"Proteins", "Significance", "Allergens"}
    section_titles = {"Abundance", r"$-\log_{10}(p_{\mathrm{adj}})$", "Allergens"}
    for text in leg.get_texts():
        if text.get_text().strip() in section_titles:
            if not text.get_text().strip() == r"$-\log_{10}(p_{\mathrm{adj}})$":
                text.set_fontsize(13)
                text.set_fontweight("bold")
            else:
                text.set_fontsize(11.5)
            # text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("Figures/volcano_plot_both_sides_labeled.png", dpi=1500, bbox_inches="tight")
    plt.show()
