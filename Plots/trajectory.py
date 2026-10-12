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

plt.rcParams['font.family'] = 'Times New Roman'


adata_with_allergens_path = "../H5AD/adata_with_allergens.h5ad"
adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)


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


def top_upregulated_allergens_cancer(adata, n=10, group="Cancer", allergen_key="byMajority"):
    res = adata.uns["rank_proteins_groups"]
    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.clip(np.array(res["pvals_adj"][group]), 1e-300, None)
    alpha = 0.05
    logfc_thresh = 0.01
    is_allergen = adata.var.loc[names, allergen_key].values == "Allergen"
    mask = (padj < alpha) & (logfc > logfc_thresh) & is_allergen
    proteins = names[mask]
    scores = logfc[mask]
    top_idx = np.argsort(scores)[-n:][::-1]
    top_proteins = proteins[top_idx]
    return top_proteins


def root_finder_healthy(adata, proteins):
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
    adata.uns["healthy_root_index"] = int(root_idx)
    return root_idx


def top_upregulated_allergens_healthy(adata, n=10, group="Cancer", allergen_key="byMajority"):
    res = adata.uns["rank_proteins_groups"]
    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.clip(np.array(res["pvals_adj"][group]), 1e-300, None)
    alpha = 0.05
    logfc_thresh = 0.01
    is_allergen = adata.var.loc[names, allergen_key].values == "Allergen"
    # 🔁 flip direction: Healthy-enriched = negative logFC in Cancer comparison
    mask = (padj < alpha) & (logfc < -logfc_thresh) & is_allergen
    proteins = names[mask]
    scores = -logfc[mask]  # stronger Healthy signal = higher score
    top_idx = np.argsort(scores)[-n:][::-1]
    return proteins[top_idx]


def print_root_signature_stats(adata, proteins, root_idx, group="Cancer"):
    res = adata.uns["rank_proteins_groups"]
    names = np.array(res["names"][group])
    logfc = np.array(res["logfoldchanges"][group])
    padj = np.array(res["pvals_adj"][group])
    idx_map = {n: i for i, n in enumerate(names)}
    print(f"\nRoot sample: {root_idx}\n")
    X = adata[:, proteins].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    for p in proteins:
        v = X[root_idx, proteins.tolist().index(p)]
        i = idx_map[p]
        print(
            f"{p}: value={v:.4f}, logFC={logfc[i]:.4f}, adj_p={padj[i]:.2e}"
        )


# Root and Proteins of Cancer Group
proteins_cancer = top_upregulated_allergens_cancer(adata_ranked_proteins, n=10)
root_cancer = root_finder_cancer(adata_ranked_proteins, proteins_cancer)
print("Top Cancer Proteins:", proteins_cancer)
print("Cancer Root ID", root_cancer)

print("="*100)

# Root and Proteins of Healthy Group
proteins_healthy = top_upregulated_allergens_healthy(adata_ranked_proteins, n=10)
root_healthy = root_finder_healthy(adata_ranked_proteins, proteins_cancer)
print("Top Healthy Proteins:", proteins_healthy)
print("Healthy Root ID", root_healthy)
exit()



def diffusion_map(adata, proteins, root_idx, label_key="label", n_neighbors=10):
    adata = adata.copy()
    adata = adata[:, proteins].copy()

    X = adata.X
    if sp.issparse(X):
        adata.X = X.toarray()

    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=min(5, len(proteins)))

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=adata.obsm["X_pca"].shape[1]
    )

    sc.tl.diffmap(adata)

    adata.uns["iroot"] = int(root_idx)
    sc.tl.dpt(adata, n_dcs=10)

    # -----------------------------
    # t-SNE instead of UMAP
    # -----------------------------
    sc.tl.tsne(adata)

    # sc.pl.tsne(
    #     adata,
    #     color="dpt_pseudotime",
    #     size=90
    # )
    #
    # sc.pl.diffmap(
    #     adata,
    #     color="dpt_pseudotime",
    #     size=90
    # )

    return adata


# adata_traj_cancer = diffusion_map(
#     adata_ranked_proteins,
#     proteins=proteins_cancer,
#     root_idx=root_cancer
# )
# adata_traj_healthy = diffusion_map(
#     adata_ranked_proteins,
#     proteins=proteins_healthy,
#     root_idx=root_healthy
# )


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


adata_traj_cancer = diffusion_map_minimal(
    adata_ranked_proteins,
    proteins=proteins_cancer,
    root_idx=root_cancer
)
adata_traj_healthy = diffusion_map_minimal(
    adata_ranked_proteins,
    proteins=proteins_healthy,
    root_idx=root_healthy
)


def plot_smooth_manifold_stream(
    adata,
    basis="tsne",
    value_key="dpt_pseudotime",
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
    y = (y - y.min()) / (y.max() - y.min())

    # -----------------------------
    # grid
    # -----------------------------
    # x_min, x_max = X[:, 0].min(), X[:, 0].max()
    # y_min, y_max = X[:, 1].min(), X[:, 1].max()

    margin_x = 0.05 * (X[:, 0].max() - X[:, 0].min())
    margin_y = 0.05 * (X[:, 1].max() - X[:, 1].min())

    x_min = X[:, 0].min() - margin_x
    x_max = X[:, 0].max() + margin_x

    y_min = X[:, 1].min() - margin_y
    y_max = X[:, 1].max() + margin_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, gridsize),
        np.linspace(y_min, y_max, gridsize)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])

    # -----------------------------
    # KDE pseudotime field
    # -----------------------------
    kde = gaussian_kde(X.T, weights=y)
    zz = kde(grid).reshape(xx.shape)

    # normalize field
    zz = (zz - zz.min()) / (zz.max() - zz.min())

    # -----------------------------
    # gradient field (flow direction)
    # -----------------------------
    gy, gx = np.gradient(zz)

    # -----------------------------
    # plot
    # -----------------------------
    plt.figure(figsize=(8, 6))

    # smooth manifold
    cf = plt.contourf(
        xx, yy, zz,
        levels=60,
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    # contour lines
    cs = plt.contour(
        xx, yy, zz,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.5
    )

    plt.clabel(cs, fontsize=11, fmt="%.2f")

    # patient points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="black",
        s=8,
        alpha=0.15
    )

    # -----------------------------
    # STREAMLINES (key change)
    # -----------------------------
    plt.streamplot(
        xx, yy,
        gx, gy,
        color="white",
        density=density,
        linewidth=1.0,
        arrowsize=1.2
    )

    # colorbar
    cbar = plt.colorbar(cf)
    cbar.set_label("Pseudotime", fontsize=15)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # plt.title("Continuous Disease Manifold with Flow Trajectories")
    plt.title("Cancer Progression Trajectory (Top Cancer-Enriched Allergens)", fontsize=14)
    # plt.title("Cancer Progression Trajectory (Top Healthy-Enriched Allergens)", fontsize=14)
    plt.axis("off")

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("Figures/disease_trajectory_based_on_top10_cancer_enriched_allergens.png", dpi=1500, bbox_inches="tight")

    plt.show()



def plot_smooth_manifold_stream2(
    adata,
    basis="tsne",
    value_key="dpt_pseudotime",
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
    y = (y - y.min()) / (y.max() - y.min())

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
    zz = (zz - zz.min()) / (zz.max() - zz.min())

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
    # patient points
    # -----------------------------
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="black",
        s=8,
        alpha=0.15
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
            zorder=10
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
    plt.title("Cancer Progression Trajectory (Top Cancer-Enriched Allergens)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_smooth_manifold_stream2(adata_traj_cancer)

