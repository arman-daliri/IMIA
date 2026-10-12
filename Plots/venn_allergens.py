import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'


# #
# # # ---------------- COLOR MAP (single source of truth) ----------------
# COLOR_MAP = {
#     "100": "#1f77b4",  # AlgPred2
#     "010": "#2ca02c",  # DeepAlgPro
#     "001": "#d62728",  # AllergenAI
#     "111": "#ff7f0e"   # Strategy-dependent selected region
# }
#
#
# def _apply_colors(v):
#     """Force correct region colors"""
#     for rid, color in COLOR_MAP.items():
#         patch = v.get_patch_by_id(rid)
#         if patch:
#             patch.set_color(color)
#             patch.set_edgecolor("black")
#
#
# def _hide_set_labels(v):
#     """Remove outer set names"""
#     for lbl in v.set_labels:
#         if lbl:
#             lbl.set_visible(False)
#
#
# def _apply_style(v, center_value, circle_values):
#     """Shared styling"""
#
#     # transparent outer circles
#     for rid in ["100", "010", "001"]:
#         patch = v.get_patch_by_id(rid)
#         if patch:
#             patch.set_alpha(0.25)
#             patch.set_edgecolor("black")
#
#     # center styling
#     center_patch = v.get_patch_by_id("111")
#     if center_patch:
#         center_patch.set_alpha(0.9)
#         center_patch.set_edgecolor("black")
#
#     # region labels
#     def set_region(rid, val):
#         lbl = v.get_label_by_id(rid)
#         if lbl:
#             lbl.set_text(f"{val}\nAllergens")
#             lbl.set_fontsize(10)
#
#     set_region("100", circle_values[0])
#     set_region("010", circle_values[1])
#     set_region("001", circle_values[2])
#
#     # center label
#     center_lbl = v.get_label_by_id("111")
#     if center_lbl:
#         center_lbl.set_text(f"{center_value}\nAllergens")
#         center_lbl.set_fontweight("bold")
#         center_lbl.set_fontsize(11)
#
#     # hide overlaps
#     for rid in ["110", "101", "011"]:
#         lbl = v.get_label_by_id(rid)
#         if lbl:
#             lbl.set_text("")

#
# def plot_full_venn_figure():
#
#     fig, axes = plt.subplots(1, 3, figsize=(24, 8))
#
#     base_sets = ("AlgPred2", "DeepAlgPro", "AllergenAI")
#     circle_vals = ("1343", "1335", "470")
#
#     # ---------------- OR ----------------
#     v1 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[0])
#     _hide_set_labels(v1)
#     _apply_colors(v1)
#     _apply_style(v1, "2165", circle_vals)
#     axes[0].set_title("By OR", fontsize=12)
#
#     # ---------------- Majority ----------------
#     v2 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[1])
#     _hide_set_labels(v2)
#     _apply_colors(v2)
#     _apply_style(v2, "996", circle_vals)
#     axes[1].set_title("By Majority", fontsize=12)
#
#     # ---------------- AND ----------------
#     v3 = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=axes[2])
#     _hide_set_labels(v3)
#     _apply_colors(v3)
#     _apply_style(v3, "202", circle_vals)
#     axes[2].set_title("By AND", fontsize=12)
#
#     # ---------------- LEGEND ----------------
#     legend_elements = [
#         Patch(facecolor=COLOR_MAP["100"], edgecolor='black', alpha=0.25, label='AlgPred2'),
#         Patch(facecolor=COLOR_MAP["010"], edgecolor='black', alpha=0.25, label='DeepAlgPro'),
#         Patch(facecolor=COLOR_MAP["001"], edgecolor='black', alpha=0.25, label='AllergenAI'),
#         Patch(facecolor=COLOR_MAP["111"], edgecolor='black', alpha=0.9,
#               label='Selected Allergens (Voting Strategy)')
#     ]
#
#     fig.legend(
#         handles=legend_elements,
#         loc='lower center',
#         ncol=4,
#         frameon=False,
#         fontsize=10
#     )
#
#     fig.subplots_adjust(bottom=0.18)
#     plt.tight_layout()
#     plt.show()
#
#
# plot_full_venn_figure()


def plot_allergen_venn():
    base_sets = ("AlgPred2", "DeepAlgPro", "AllergenAI")
    circle_vals = ("1343", "1335", "470")

    COLOR_MAP = {
        "100": "#1f77b4",
        "010": "#2ca02c",
        "001": "#d62728",
        "111": "#ff7f0e"
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    v = venn3((1, 1, 1, 1, 1, 1, 1), base_sets, ax=ax)

    # ---------------- OUTER LABELS ----------------
    v.set_labels[0].set_text("AlgPred2.0")
    v.set_labels[0].set_fontsize(15)
    v.set_labels[1].set_text("DeepAlgPro")
    v.set_labels[1].set_fontsize(15)
    v.set_labels[2].set_text("AllergenAI")
    v.set_labels[2].set_fontsize(15)

    # ---------------- COLORS ----------------
    for rid in ["100", "010", "001"]:
        patch = v.get_patch_by_id(rid)
        if patch:
            patch.set_color(COLOR_MAP[rid])
            patch.set_alpha(0.25)
            patch.set_edgecolor("black")

    for rid in ["110", "101", "011", "111"]:
        patch = v.get_patch_by_id(rid)
        if patch:
            patch.set_facecolor("#ff7f0e")  # same orange for all overlaps
            patch.set_edgecolor("none")  # remove stroke
            patch.set_linewidth(0)
            patch.set_antialiased(False)
            patch.set_zorder(3)

    # ---------------- IMPORTANT: CLEAR ALL REGION LABELS ----------------
    for rid in ["100", "010", "001", "110", "101", "011", "111"]:
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text("")  # remove default "1"

    # ---------------- INNER CIRCLE LABELS ----------------
    region_values = {
        "100": f"{circle_vals[0]}\nTotal Allergens",
        "010": f"{circle_vals[1]}\nTotal Allergens",
        "001": f"{circle_vals[2]}\nTotal Allergens",
    }
    # region_values = {
    #     "100": f"{circle_vals[0]}\nAllergens",
    #     "010": f"{circle_vals[1]}\nAllergens",
    #     "001": f"{circle_vals[2]}\nAllergens",
    # }

    for rid, val in region_values.items():
        lbl = v.get_label_by_id(rid)
        if lbl:
            lbl.set_text(val)
            lbl.set_fontsize(12)

    # ---------------- CENTER OVERLAP ----------------
    center = v.get_label_by_id("111")
    if center:
        center.set_text("996\nAllergens")
        center.set_fontsize(18)
        center.set_fontweight("bold")
        center.set_ha("center")
        center.set_va("center")

    ax.set_title("Majority Voting", fontsize=20)
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "Figures/allergens_by_majority.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()


plot_allergen_venn()

