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
import scipy.sparse as sp_sparse
from scipy.ndimage import gaussian_filter1d



plt.rcParams['font.family'] = 'Times New Roman'


adata_ranked_proteins_path = "../H5AD/adata_ranked_proteins.h5ad"
adata_ranked_proteins = sc.read_h5ad(adata_ranked_proteins_path)
print(adata_ranked_proteins.obs_names)

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
    if sp_sparse.issparse(X):
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


def diffusion_map_minimal2(
    adata,
    proteins,
    root_idx,
    n_neighbors=10,
    n_dcs=7,
    make_tsne=True,
    random_state=42
):
    adata = adata.copy()

    np.random.seed(random_state)

    # -----------------------------
    # 1. subset proteins
    # -----------------------------
    adata = adata[:, proteins].copy()

    # X = adata.X
    # if sp.sparse.issparse(X):
    #     X = X.toarray()
    #     adata.X = X

    # -----------------------------
    # 2. neighbors graph
    # -----------------------------
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
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

    # -----------------------------
    # 6. extract ordering
    # -----------------------------
    order_df = adata.obs[["dpt_pseudotime"]].copy()

    ordered_cases = (
        order_df.sort_values("dpt_pseudotime")
        .index
        .tolist()
    )

    return adata, ordered_cases


def plot_protein_trajectory(adata, ordered_cases, protein_name):
    # -----------------------------
    # 1. extract expression vector
    # -----------------------------
    X = adata[:, protein_name].X

    if sp.issparse(X):
        X = X.toarray().flatten()
    else:
        X = np.asarray(X).flatten()

    # -----------------------------
    # 2. map expression to ordered cases
    # -----------------------------
    # align expression with obs_names
    expr_map = dict(zip(adata.obs_names, X))

    ordered_expr = np.array([expr_map[c] for c in ordered_cases])

    # -----------------------------
    # 3. plot
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(ordered_expr) + 1), ordered_expr, linewidth=2)

    plt.xlabel("Pseudotime-ordered samples (1 → 360)")
    plt.ylabel(f"Abundance: {protein_name}")
    plt.title(f"Trajectory of {protein_name}")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return ordered_expr


def plot_protein_trajectories_multi(adata, ordered_cases, protein_list):
    X = adata[:, protein_list].X

    if sp.issparse(X):
        X = X.toarray()

    # shape: (n_samples, n_proteins)
    X = np.asarray(X)

    # -----------------------------
    # 2. align samples to ordering
    # -----------------------------
    sample_index = {name: i for i, name in enumerate(adata.obs_names)}
    ordered_idx = [sample_index[c] for c in ordered_cases]

    X_ordered = X[ordered_idx, :]  # (n_samples, n_proteins)

    # -----------------------------
    # 3. plot
    # -----------------------------
    plt.figure(figsize=(12, 5))

    x_axis = np.arange(1, len(ordered_cases) + 1)

    for j, protein in enumerate(protein_list):
        plt.plot(
            x_axis,
            X_ordered[:, j],
            linewidth=2,
            label=protein
        )

    plt.xlabel("Pseudotime-ordered samples (1 → N)")
    plt.ylabel("Abundance")
    plt.title("Protein trajectories along pseudotime")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return X_ordered


def plot_smoothed_trajectories(
    adata,
    ordered_cases,
    protein_list,
    sigma=5
):
    X = adata[:, protein_list].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    idx = {n: i for i, n in enumerate(adata.obs_names)}
    ordered_idx = [idx[c] for c in ordered_cases]

    X = X[ordered_idx, :]

    plt.figure(figsize=(12, 5))
    x = np.arange(1, len(ordered_cases) + 1)

    for j, protein in enumerate(protein_list):
        smooth = gaussian_filter1d(X[:, j], sigma=sigma)
        plt.plot(x, smooth, linewidth=2, label=protein)

    plt.xlabel("Pseudotime order")
    plt.ylabel("Smoothed abundance")
    plt.title("Smoothed protein trajectories (Gaussian filter)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_trajectory(adata, ordered_cases, protein_list=None, smooth_sigma=5):
    if protein_list is None:
        protein_list = adata.var_names

    X = adata[:, protein_list].X

    if sp.issparse(X):
        X = X.toarray()

    X = np.asarray(X)

    # -----------------------------
    # 2. reorder samples
    # -----------------------------
    idx = {name: i for i, name in enumerate(adata.obs_names)}
    ordered_idx = [idx[c] for c in ordered_cases]

    X = X[ordered_idx, :]  # (samples × proteins)

    # -----------------------------
    # 3. compute mean trajectory
    # -----------------------------
    mean_traj = np.nanmean(X, axis=1)

    # -----------------------------
    # 4. optional smoothing
    # -----------------------------
    mean_smooth = gaussian_filter1d(mean_traj, sigma=smooth_sigma)

    # -----------------------------
    # 5. plot
    # -----------------------------
    x_axis = np.arange(1, len(ordered_cases) + 1)

    plt.figure(figsize=(10, 4))
    # plt.plot(x_axis, mean_traj, alpha=0.4, label="Raw mean")
    plt.plot(x_axis, mean_smooth, linewidth=2, label="Smoothed mean")

    plt.xlabel("Pseudotime order")
    plt.ylabel("Average abundance")
    plt.title("Consensus protein trajectory")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return mean_traj, mean_smooth


total_proteins = top_upregulated_allergens + top_downregulated_allergens
_adata, ordered_cases = diffusion_map_minimal2(
    adata_ranked_proteins,
    total_proteins,
    root_healthy_max
)

# plot_protein_trajectory(_adata, ordered_cases, "D1BP93_VEIPT")
plot_average_trajectory(_adata, ordered_cases, total_proteins)



# total_proteins = top_upregulated_allergens + top_downregulated_allergens
# # total_proteins = top_upregulated_allergens
#
#
# adata_traj_cancer = diffusion_map_minimal(
#     adata_ranked_proteins,
#     proteins=total_proteins,
#     root_idx=root_healthy_max,
#     n_dcs=10,
# )
# # adata_traj_healthy = diffusion_map_minimal(
# #     adata_ranked_proteins,
# #     proteins=proteins_healthy,
# #     root_idx=root_healthy
# # )
#
#
# plot_smooth_manifold_stream3(adata_traj_cancer)

