import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
})


def plot_pimms_loss(
    zjc_path="./Temp/zjc_pimms.csv",
    multi_path="./Temp/multi_center_pimms.csv"
):
    zjc = pd.read_csv(zjc_path, delimiter="\t")
    multi = pd.read_csv(multi_path, delimiter="\t")

    palette = ["#6B8E23", "#7B2CBF"]

    plt.figure(figsize=(7, 5))

    plt.plot(
        zjc["epoch"],
        zjc["train_loss"],
        label="ZJC cohort",
        linewidth=2,
        color=palette[0]
    )

    plt.plot(
        multi["epoch"],
        multi["train_loss"],
        label="WZ/ZJT/SCC multi-center cohort",
        linewidth=2,
        color=palette[1]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("VAE Training Loss Curves Across Cohorts (PIMMS AETransformer)")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "Figures/pimms_training_loss.png",
        dpi=1500,
        bbox_inches="tight"
    )
    plt.show()


plot_pimms_loss()
