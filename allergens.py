import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch


def plot_majority_center_highlight():
    """
    Venn diagram where only the triple intersection is highlighted.
    All other regions share the same color.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    v = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=("AlgPred2", "DeepAlgPro", "AllergenAI"),
        ax=ax
    )

    # --- base color for all regions ---
    base_color = "#d0d0d0"
    highlight_color = "#ff7f0e"  # orange highlight for center

    # all regions except center
    for region_id in ["100", "010", "001", "110", "101", "011"]:
        patch = v.get_patch_by_id(region_id)
        if patch:
            patch.set_color(base_color)
            patch.set_alpha(1.0)
            patch.set_edgecolor("white")

    # center region (111)
    center_patch = v.get_patch_by_id("111")
    if center_patch:
        center_patch.set_color(highlight_color)
        center_patch.set_alpha(1.0)
        center_patch.set_edgecolor("black")

    # label only center if needed
    for region_id in ["100", "010", "001", "110", "101", "011"]:
        lbl = v.get_label_by_id(region_id)
        if lbl:
            lbl.set_text("")

    v.get_label_by_id("111").set_text("996")

    ax.set_title("Majority Voting (Center Highlight)", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_majority_center_highlight2():
    """
    Venn diagram:
    - transparent circles
    - center highlighted
    - numbers + 'Allergens' inside circles
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    v = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=("AlgPred2", "DeepAlgPro", "AllergenAI"),
        ax=ax
    )

    # ---------------- transparent circles ----------------
    for region_id in ["100", "010", "001"]:
        patch = v.get_patch_by_id(region_id)
        if patch:
            patch.set_alpha(0.25)
            patch.set_edgecolor("black")

    # ---------------- center highlight ----------------
    center_patch = v.get_patch_by_id("111")
    if center_patch:
        center_patch.set_color("#ff7f0e")
        center_patch.set_alpha(0.9)
        center_patch.set_edgecolor("black")

    # ---------------- helper for number + label ----------------
    def set_region_text(region_id, number):
        lbl = v.get_label_by_id(region_id)
        if lbl:
            lbl.set_text(f"{number}\nAllergens")
            lbl.set_fontsize(10)

    # set circle values
    set_region_text("100", "1343")
    set_region_text("010", "1335")
    set_region_text("001", "470")

    # center overlap
    center_lbl = v.get_label_by_id("111")
    if center_lbl:
        center_lbl.set_text("996\nAllergens")
        center_lbl.set_fontweight("bold")
        center_lbl.set_fontsize(11)

    # hide other overlaps
    for rid in ["110", "101", "011"]:
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text("")

    ax.set_title("Majority Voting (Center Highlight)", fontsize=12)

    plt.tight_layout()
    plt.show()


# plot_majority_center_highlight()


def _apply_style(v, center_value, circle_values):
    """
    Shared styling for all 3 plots
    """

    # ---------------- transparent circles ----------------
    for region_id in ["100", "010", "001"]:
        patch = v.get_patch_by_id(region_id)
        if patch:
            patch.set_alpha(0.25)
            patch.set_edgecolor("black")

    # ---------------- center highlight ----------------
    center_patch = v.get_patch_by_id("111")
    if center_patch:
        center_patch.set_color("#ff7f0e")
        center_patch.set_alpha(0.9)
        center_patch.set_edgecolor("black")

    # ---------------- circle values ----------------
    def set_region(rid, val):
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text(f"{val}\nAllergens")
            lbl.set_fontsize(10)

    set_region("100", circle_values[0])
    set_region("010", circle_values[1])
    set_region("001", circle_values[2])

    # ---------------- center ----------------
    center_lbl = v.get_label_by_id("111")
    if center_lbl:
        center_lbl.set_text(f"{center_value}\nAllergens")
        center_lbl.set_fontweight("bold")
        center_lbl.set_fontsize(11)

    # hide other overlaps
    for rid in ["110", "101", "011"]:
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text("")


def plot_full_venn_figure():
    """
    Single function: creates full 3-panel Venn figure
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    base_sets = ("AlgPred2", "DeepAlgPro", "AllergenAI")
    circle_vals = ("1343", "1335", "470")

    # ---------------- OR (LEFT) ----------------
    v1 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[0])
    _apply_style(v1, "2165", circle_vals)
    axes[0].set_title("By OR", fontsize=12)

    # ---------------- Majority (MIDDLE) ----------------
    v2 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[1])
    _apply_style(v2, "996", circle_vals)
    axes[1].set_title("By Majority", fontsize=12)

    # ---------------- AND (RIGHT) ----------------
    v3 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[2])
    _apply_style(v3, "202", circle_vals)
    axes[2].set_title("By AND", fontsize=12)

    plt.tight_layout()
    plt.show()


plot_full_venn_figure()
csv_file = "allergens.csv"

# Load CSV
# df = pd.read_csv(csv_file)
#
# # Columns of interest
# cols = ["AlgPred2", "DeepAlgPro", "AllergenAI", "byAND", "byOR", "byMajority"]
#
# # 1. Print unique values per column
# print("=== Unique values per column ===\n")
# for col in cols:
#     uniques = df[col].dropna().unique()
#     print(f"{col}: {uniques}")
#
# # 2. Count occurrences per class in each column
# print("\n=== Class distributions ===\n")
# for col in cols:
#     print(f"\n{col}:")
#     counts = df[col].value_counts(dropna=False)
#     print(counts)


def count_two_model_disagreement_with_nan(df):
    """
    Counts rows where:
    - exactly two models have non-NaN values
    - the third is NaN
    - the two available predictions disagree
    """
    cols = ["AlgPred2", "DeepAlgPro", "AllergenAI"]
    count = 0
    for _, row in df[cols].iterrows():
        values = row.values

        # keep only non-NaN values
        non_nan = [v for v in values if pd.notna(v)]

        # exactly two predictions available
        if len(non_nan) == 2:
            # disagreement condition
            if non_nan[0] != non_nan[1]:
                count += 1
    print(count)



def analyze_byAND_allergen(df):
    # A0A096AIG6_9BACT Allergen        NaN        NaN
    # A0A096AWQ7_9BACT Allergen   Allergen   Allergen
    cols = ["Accession", "AlgPred2", "DeepAlgPro", "AllergenAI"]
    subset = df[df["byAND"] == "Allergen"]
    print("Number of byAND == Allergen:", len(subset))
    print("\nRows (only model columns):\n")
    print(subset[cols].to_string(index=False))


def analyze_byMajority_allergen(df):
    # A0A096AIG6_9BACT Allergen        NaN        NaN
    # A0A096AWQ7_9BACT Allergen   Allergen   Allergen
    cols = ["Accession", "AlgPred2", "DeepAlgPro", "AllergenAI"]
    subset = df[df["byOR"] == "Allergen"]
    print("Number of byAND == Allergen:", len(subset))
    print("\nRows (only model columns):\n")
    print(subset[cols].to_string(index=False))


def analyze_byMajority_allergen(df):
    # A0A096AIG6_9BACT Allergen        NaN        NaN
    # A0A096AWQ7_9BACT Allergen   Allergen   Allergen
    cols = ["Accession", "AlgPred2", "DeepAlgPro", "AllergenAI"]
    subset = df[df["byMajority"] == "Allergen"]
    print("Number of byAND == Allergen:", len(subset))
    print("\nRows (only model columns):\n")
    print(subset[cols].to_string(index=False))


# df = pd.read_csv(csv_file)
# analyze_byMajority_allergen(df)


# def compare_csv_with_adata_fixed(csv_path, adata):
#     cols = ["AlgPred2", "DeepAlgPro", "AllergenAI", "byAND", "byOR", "byMajority"]
#     df = pd.read_csv(csv_path).set_index("Accession")
#
#     common = df.index.intersection(adata.var_names)
#
#     print("Common proteins:", len(common))
#     print("CSV only:", len(df.index.difference(adata.var_names)))
#     print("AnnData only:", len(adata.var_names.difference(df.index)))
#
#     df_common = df.loc[common, cols]
#     adata_common = adata.var.loc[common, cols]
#
#     mismatches = {}
#
#     for col in cols:
#
#         a = df_common[col]
#         b = adata_common[col]
#
#         # mismatch only when BOTH are NOT NaN and different
#         mask = a.notna() & b.notna() & (a != b)
#
#         mismatches[col] = mask.sum()
#
#         print(f"\nColumn: {col}")
#         print("True mismatches:", mask.sum())
#
#         if mask.sum() > 0:
#             print("\nExample mismatches:")
#             print(pd.DataFrame({
#                 "CSV": a[mask],
#                 "AnnData": b[mask]
#             }).head())
#     return mismatches
#
#
# mismatches = compare_csv_with_adata_fixed(csv_file, adata_ranked_proteins)
# print(mismatches)


# def summarize_adata_var(adata):
#     cols = ["AlgPred2", "DeepAlgPro", "AllergenAI", "byAND", "byOR", "byMajority"]
#     var_df = adata.var
#
#     # 1. Unique values per column
#     print("=== Unique values per column (AnnData.var) ===\n")
#     for col in cols:
#         uniques = var_df[col].dropna().unique()
#         print(f"{col}: {uniques}")
#
#     # 2. Value counts per column
#     print("\n=== Class distributions (AnnData.var) ===\n")
#     for col in cols:
#         print(f"\n{col}:")
#         counts = var_df[col].value_counts(dropna=False)
#         print(counts)



# (360, 3790)
# === Unique values per column ===
#
# AlgPred2: ['Non-Allergen' 'Allergen']
# DeepAlgPro: ['Allergen' 'Non-Allergen']
# AllergenAI: ['Non-Allergen' 'Allergen']
# byAND: ['Non-Allergen' 'Allergen']
# byOR: ['Allergen' 'Non-Allergen']
# byMajority: ['Non-Allergen' 'Allergen' 'Likely Allergen']
#
# === Class distributions ===
#
#
# AlgPred2:
# AlgPred2
# Non-Allergen    2437
# Allergen        1343
# NaN               10
# Name: count, dtype: int64
#
# DeepAlgPro:
# DeepAlgPro
# Non-Allergen    2267
# Allergen        1335
# NaN              188
# Name: count, dtype: int64
#
# AllergenAI:
# AllergenAI
# Non-Allergen    3141
# Allergen         470
# NaN              179
# Name: count, dtype: int64
#
# byAND:
# byAND
# Non-Allergen    3588
# Allergen         202
# Name: count, dtype: int64
#
# byOR:
# byOR
# Allergen        2165
# Non-Allergen    1625
# Name: count, dtype: int64
#
# byMajority:
# byMajority
# Non-Allergen       2793
# Allergen            996
# Likely Allergen       1
# Name: count, dtype: int64
