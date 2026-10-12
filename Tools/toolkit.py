from os.path import exists
from os.path import join
import pandas as pd
import scanpy as sc
import numpy as np
import anndata
import json
from scipy.stats import fisher_exact


# csv_allergens_file = "allergens.csv"
# adata_original_path = "Temp/adata_final.h5ad"

def adata_head(adata, name):
    _x = adata.X
    print(f"=== {name} ===")
    print(f"Shape: {_x.shape}")
    print(f"Mean: {np.mean(_x):.4f}")
    print(f"Min:  {np.min(_x):.4f}")
    print(f"Max:  {np.max(_x):.4f}")
    print(f"Non-zero fraction: {(np.count_nonzero(_x) / _x.size):.4f}")
    print(adata)
    print()


def clean_allergen_csv_file():
    df = pd.read_csv("../Temp/allergens.csv")
    df = df.iloc[:, 1:]
    df.columns.values[0] = "Accession"
    df.to_csv("allergens.csv", index=False)
    print("First column removed and saved to new_file.csv")


def report_allergens_csv_file():
    csv_allergens_path = "../allergens.csv"
    csv_allergens_df = pd.read_csv(csv_allergens_path)
    print("Shape (rows, columns):", csv_allergens_df.shape)
    print("\nHeader (column names):")
    print(list(csv_allergens_df.columns))
    print("\nSchema (column name : dtype):")
    print(csv_allergens_df.dtypes)
    print(csv_allergens_df.head())


def compare_allergen_csv_with_adata(csv_path, adata):
    df = pd.read_csv(csv_path)
    csv_proteins = set(df["Accession"].astype(str))
    adata_proteins = set(adata.var_names.astype(str))
    overlap = csv_proteins & adata_proteins
    print(f"Proteins in CSV: {len(csv_proteins)}")
    print(f"Proteins in AnnData: {len(adata_proteins)}")
    print(f"Overlap: {len(overlap)}")
    return overlap


def add_allergens_to_adata(csv_path, adata):
    new_adata = adata.copy()
    df = pd.read_csv(csv_path).set_index("Accession")
    cols = ["AlgPred2", "DeepAlgPro", "AllergenAI", "byAND", "byOR", "byMajority"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    aligned = df.loc[new_adata.var_names, cols]
    aligned = aligned.replace({"": pd.NA, "None": pd.NA, "none": pd.NA})
    for c in cols:
        new_adata.var[c] = aligned[c].astype("category")
    if "allergen_label" in new_adata.var.columns:
        new_adata.var.drop(columns=["allergen_label"], inplace=True)
    return new_adata


def report_allergens_adata(adata):
    total_vars = adata.n_vars
    num_allergens = (adata.var['byMajority'] == 'Allergen').sum()
    print(f"Total variables: {total_vars}")
    print(f"Allergen variables: {num_allergens}")


def save_new_adata_with_allergens():
    csv_allergens_path = "../allergens.csv"
    adata_original_path = "../H5AD/adata_final.h5ad"

    adata_microbial = sc.read_h5ad(adata_original_path)
    adata_head(adata_microbial, name="Adata - Without Allergens")

    adata_with_allergens = add_allergens_to_adata(csv_allergens_path, adata_microbial)
    adata_head(adata_with_allergens, name="Adata - Added Allergens")

    report_allergens_adata(adata_with_allergens)
    # Save fixed adata_with_allergens
    sc.write("../H5AD/adata_with_allergens.h5ad", adata_with_allergens)
    print("Done Saving Temp")


def top_features_with_stats(adata, top_n=20):
    result = adata.uns['rank_proteins_groups']
    groups = result['names'].dtype.names
    for group in groups:
        print(f"\nTop {top_n} features for group '{group}':")
        print(f"{'Feature':<15} {'LogFC':>8} {'Score':>8} {'Adj Pval':>12}")
        print("-"*50)
        for i in range(top_n):
            name = result['names'][group][i]
            logfc = result['logfoldchanges'][group][i]
            score = result['scores'][group][i]
            pval_adj = result['pvals_adj'][group][i]
            print(f"{name:<15} {logfc:8.3f} {score:8.3f} {pval_adj:12.3e}")


def report_allergen_enrichment(adata, group='Cancer', allergen_col='byMajority', alpha=0.05):
    result = adata.uns['rank_proteins_groups']
    names = result['names'][group]
    pvals_adj = result['pvals_adj'][group]

    names = np.array(names)
    pvals_adj = np.array(pvals_adj)

    sig_mask = pvals_adj < alpha
    sig_proteins = names[sig_mask]

    allergen_mask = adata.var[allergen_col] == 'Allergen'
    allergen_set = set(adata.var_names[allergen_mask])

    a = np.sum([p in allergen_set for p in sig_proteins])
    b = len(allergen_set) - a
    c = len(sig_proteins) - a
    d = len(adata.var_names) - (a + b + c)

    table = np.array([[a, b], [c, d]])
    oddsratio, pvalue = fisher_exact(table)

    print("Enrichment of allergens among significant proteins")
    print(pd.DataFrame(table,
          index=['Allergen','Non-Allergen'],
          columns=['Significant','Not significant']))

    print(f"\nOdds ratio: {oddsratio:.3f}")
    print(f"P-value: {pvalue:.3e}")

# print(adata_microbial)
# adata_head(adata_microbial, "Adata Final")
# print(adata_microbial.var["taxid"])
