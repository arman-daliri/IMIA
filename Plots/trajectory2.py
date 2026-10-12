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
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Times New Roman'


adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)

# Root and Proteins of Cancer Group
top_upregulated_allergens = [
    "D1BP93_VEIPT",
    "F3B2V5_9FIRM",
    "A0A2W5KWT7_9CORY",
    "A0A2S7ZLE7_9FIRM",
    "W1VMM1_STRPA",
    "D2ZSM3_NEIMU",
    "A0A2T4TA75_9FIRM",
    "H1HWB9_9FIRM",
    "A0A1E9MPU5_9NEIS",
    "K0XE54_9FIRM"
]

top_downregulated_allergens = [
    "A0A239YF41_9FIRM",
    "E6KWI6_9PAST",
    "D6GSY8_FILAD",
    "X8ISL9_9STRE",
    "A0A2K9HAQ2_9BACT",
    "A0A6L5DSP1_9BACT",
    "UPI000469B5B7",
    "A0A3N2MZC8_9BACT",
    "A0A379EY57_9BACT",
    "L1MCS5_9BACT"
]


def root_finder_cancer(adata, proteins):
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_samples = X.shape[0]
    rank_matrix = np.zeros((n_samples, len(proteins)))
    for j in range(len(proteins)):
        values = X[:, j]
        ranks = np.argsort(np.argsort(values)) + 1
        rank_matrix[:, j] = ranks
    rank_sum = rank_matrix.sum(axis=1)
    root_idx = np.argmin(rank_sum)
    adata.uns["consensus_root_index"] = int(root_idx)
    return root_idx


def root_finder_healthy_max(adata, proteins):
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_samples = X.shape[0]
    rank_matrix = np.zeros((n_samples, len(proteins)))
    for j in range(len(proteins)):
        values = X[:, j]
        # rank: low → high expression
        ranks = np.argsort(np.argsort(values)) + 1
        rank_matrix[:, j] = ranks
    rank_sum = rank_matrix.sum(axis=1)
    # -----------------------------
    # KEY CHANGE: Healthy root
    # -----------------------------
    root_idx = np.argmax(rank_sum)
    # root_idx = np.argmin(rank_sum)
    adata.uns["healthy_root_index"] = int(root_idx)
    return root_idx


def root_finder_healthy_least(adata, proteins):
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_samples = X.shape[0]
    rank_matrix = np.zeros((n_samples, len(proteins)))
    for j in range(len(proteins)):
        values = X[:, j]
        # rank: low → high expression
        ranks = np.argsort(np.argsort(values)) + 1
        rank_matrix[:, j] = ranks
    rank_sum = rank_matrix.sum(axis=1)
    # -----------------------------
    # KEY CHANGE: Healthy root
    # -----------------------------
    root_idx = np.argmax(rank_sum)
    # root_idx = np.argmin(rank_sum)
    adata.uns["healthy_root_index"] = int(root_idx)
    return root_idx


root_cancer = root_finder_cancer(adata_ranked_proteins, top_upregulated_allergens)
root_healthy_max = root_finder_healthy_max(adata_ranked_proteins, top_downregulated_allergens)
root_healthy_least = root_finder_healthy_max(adata_ranked_proteins, top_downregulated_allergens)

print("Top Upregulated Allergens:", top_upregulated_allergens)
print("Least Cancer-like root", root_cancer)
print("="*100)
print("Top Downregulated Allergens:", top_downregulated_allergens)
print("Max healthy-like root", root_healthy_max)
print("Least healthy-like root", root_healthy_least)
print("="*100)


def diffusion_map_minimal(
    adata,
    proteins,
    root_idx,
    n_neighbors=10,
    n_dcs=7,
    make_tsne=True,
    random_state=42
):
    adata = adata.copy()

    # reproducibility
    np.random.seed(random_state)

    # -----------------------------
    # 1. subset proteins
    # -----------------------------
    adata = adata[:, proteins].copy()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
        adata.X = X

    # -----------------------------
    # 2. neighbors graph
    # -----------------------------
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=None,
        random_state=random_state
    )

    # -----------------------------
    # 3. diffusion map
    # -----------------------------
    sc.tl.diffmap(adata)

    # -----------------------------
    # 4. diffusion pseudotime
    # -----------------------------
    adata.uns["iroot"] = int(root_idx)

    sc.tl.dpt(
        adata,
        n_dcs=n_dcs
    )

    # -----------------------------
    # 5. t-SNE
    # -----------------------------
    if make_tsne:
        sc.tl.tsne(
            adata,
            random_state=random_state
        )

    return adata


# adata_traj_cancer = diffusion_map_minimal(
#     adata_ranked_proteins,
#     proteins=proteins_cancer,
#     root_idx=root_cancer
# )
# adata_traj_healthy = diffusion_map_minimal(
#     adata_ranked_proteins,
#     proteins=proteins_healthy,
#     root_idx=root_healthy
# )


def plot_smooth_manifold_stream3(
    adata,
    basis="tsne",
    value_key="dpt_pseudotime",
    group_key="label",
    gridsize=500,
    cmap="viridis",
    density=0.3
):

    # -----------------------------
    # embedding
    # -----------------------------
    if basis == "diffmap":
        X = adata.obsm["X_diffmap"][:, :2]
    else:
        X = adata.obsm[f"X_{basis}"]

    y = adata.obs[value_key].values.astype(float)

    # normalize pseudotime
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # -----------------------------
    # grid with margin
    # -----------------------------
    margin_x = 0.05 * (X[:, 0].max() - X[:, 0].min())
    margin_y = 0.05 * (X[:, 1].max() - X[:, 1].min())

    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, gridsize),
        np.linspace(y_min, y_max, gridsize)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])

    # -----------------------------
    # KDE field
    # -----------------------------
    kde = gaussian_kde(X.T, weights=y)
    zz = kde(grid).reshape(xx.shape)

    # normalize field
    zz = (zz - zz.min()) / (zz.max() - zz.min() + 1e-12)

    # -----------------------------
    # gradient (flow)
    # -----------------------------
    gy, gx = np.gradient(zz)

    # -----------------------------
    # plot
    # -----------------------------
    plt.figure(figsize=(8, 6))

    cf = plt.contourf(
        xx, yy, zz,
        levels=60,
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    cs = plt.contour(
        xx, yy, zz,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.5
    )
    plt.clabel(cs, fontsize=10, fmt="%.2f")

    # -----------------------------
    # patient points (Cancer vs Healthy)
    # -----------------------------
    groups = adata.obs[group_key].values

    cancer_mask = groups == "Cancer"
    healthy_mask = groups == "Healthy"

    plt.scatter(
        X[healthy_mask, 0],
        X[healthy_mask, 1],
        c="#1f77b4",
        s=15,
        alpha=0.4,
        # marker="o",
        label="Healthy"
    )

    plt.scatter(
        X[cancer_mask, 0],
        X[cancer_mask, 1],
        c="#ff7f0e",
        s=15,
        alpha=0.4,
        # marker="p",
        label="Cancer"
    )

    # -----------------------------
    # streamlines (trajectory flow)
    # -----------------------------
    plt.streamplot(
        xx, yy,
        gx, gy,
        color="white",
        density=density,
        linewidth=1.0,
        arrowsize=1.2
    )

    # -----------------------------
    # ROOT MARKER
    # -----------------------------
    root_idx = adata.uns.get("iroot", None)

    if root_idx is not None:
        root_x = X[root_idx, 0]
        root_y = X[root_idx, 1]

        plt.scatter(
            root_x,
            root_y,
            c="red",
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=1.2,
            zorder=10,
            label="Root"
        )

        plt.text(
            root_x,
            root_y,
            " Root",
            color="red",
            fontsize=11,
            weight="bold"
        )

    # -----------------------------
    # colorbar
    # -----------------------------
    cbar = plt.colorbar(cf)
    cbar.set_label("Pseudotime", fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # -----------------------------
    # styling
    # -----------------------------
    plt.title("Cancer Progression Trajectory (Diffusion Map & Pseudotime)", fontsize=13)
    plt.axis("off")

    save_path = "/Users/pedram/Desktop/Project/Figures/disease_trajectory_fixed.png"
    plt.savefig(save_path, dpi=1500, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


total_proteins = top_upregulated_allergens + top_downregulated_allergens
# total_proteins = top_upregulated_allergens


adata_traj_cancer = diffusion_map_minimal(
    adata_ranked_proteins,
    proteins=total_proteins,
    root_idx=root_healthy_max,
    n_dcs=10,
)
# adata_traj_healthy = diffusion_map_minimal(
#     adata_ranked_proteins,
#     proteins=proteins_healthy,
#     root_idx=root_healthy
# )


plot_smooth_manifold_stream3(adata_traj_cancer)

