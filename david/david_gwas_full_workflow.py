import os
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from limix_lmm import LMM
from matplotlib.pyplot import title
from pandas_plink import read_plink, write_plink1_bin
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# --- Compatibility patch for SciPy deprecations (Colab / Python 3.12) ---
# Some older packages such as `limix` still reference deprecated SciPy functions
# like `scipy.log`, `scipy.exp`, and `scipy.sqrt`. These were removed starting from
# SciPy 1.12 (2024) and are no longer available in recent Python environments.
#
# On Google Colab, the default Python version is currently 3.12, which only supports
# SciPy >= 1.12. Older versions of SciPy (<= 1.10.1), which still provided these
# functions, cannot be installed because they require Python < 3.12.
#
# Therefore, we manually alias these missing functions to their NumPy equivalents.
# This "monkey patch" allows legacy code (e.g., `limix`) to run under the current
# Colab setup without downgrading Python or SciPy.
# -----------------------------------------------------------------------------

if not hasattr(scipy, 'dot'):
    scipy.dot = np.dot
if not hasattr(scipy, 'einsum'):
    scipy.einsum = np.einsum
if not hasattr(scipy, 'log'):
    scipy.log = np.log
if not hasattr(scipy, 'sign'):
    scipy.sign = np.sign
if not hasattr(scipy, 'sqrt'):
    scipy.sqrt = np.sqrt

os.makedirs("data/preprocessing", exist_ok=True)

subprocess.run([r"plink_win64_20250819/plink.exe",
                "--bfile", r"../ALL.chr22_GRCh38.genotypes.20170504/ALL.chr22_GRCh38.genotypes.20170504",
                "--snps-only",
                "--geno", "0.02",
                "--maf", "0.01",
                "--hwe", "1e-6",
                "--make-bed",
                "--out", r"data/preprocessing/chr22_step1_common"])

subprocess.run([r"plink_win64_20250819/plink.exe",
                "--bfile", r"data/preprocessing/chr22_step1_common",
                "--mind", "0.02",
                "--make-bed",
                "--out", r"data/preprocessing/chr22_preprocessed"])

subprocess.run([r"plink_win64_20250819/plink.exe",
                "--bfile", r"../ALL.chr22_GRCh38.genotypes.20170504/ALL.chr22_GRCh38.genotypes.20170504",
                "--indep-pairwise", "200", "50", "0.2",
                "--out", r"data/preprocessing/chr22_pruned"])

subprocess.run([r"plink_win64_20250819/plink.exe",
                "--bfile", r"../ALL.chr22_GRCh38.genotypes.20170504/ALL.chr22_GRCh38.genotypes.20170504",
                "--extract", r"data/preprocessing/chr22_pruned.prune.in",
                "--pca", "10",
                "--out", r"data/preprocessing/chr22_pca10"])

eigenvec = r"data/preprocessing/chr22_pca10.eigenvec"

df = pd.read_csv(eigenvec, sep=r"\s+", header=None)
ncols = df.shape[1]
pc_cols = list(range(2, ncols))  # columns 2..end are PCs

# z-score each PC across individuals
pcs = df.iloc[:, pc_cols]
pcs_z = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0, ddof=0)

df_z = df.copy()
df_z.iloc[:, pc_cols] = pcs_z

df_z.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, len(pc_cols)+1)]

out = eigenvec.replace(".eigenvec", ".eigenvec.zscore")
df_z.to_csv(out, sep="\t", index=False)

# load genetic data
bfile = r'data/preprocessing/chr22_preprocessed'
bim, fam, G = read_plink(bfile)

X_real = G.compute().T

rng = np.random.default_rng(42)

def simulate_pheno(X, idx_caus, var_expl, direction=None):
    # Ensure that the number of causal variant indices matches the number of variances explained.
    assert len(idx_caus) == len(var_expl)

    # If no direction is provided, randomly assign a positive or negative direction for each causal variant.
    if direction is None:
        direction = 2. * (rng.random(len(idx_caus)) > 0.5) - 1.
    # Ensure that the number of directions matches the number of causal variant indices.
    assert len(idx_caus) == len(direction)

    # Compute the remaining variance after accounting for the variance explained by the causal variants.
    ve = 1 - var_expl.sum()
    # Ensure that the total variance explained by causal variants is less than 1.
    assert ve > 0, 'sum(var_expl) should be < 1'

    # Compute the effect sizes for the causal variants based on the variance they explain and their direction.
    beta = np.sqrt(var_expl) * direction

    # Extract the columns of X corresponding to the causal variants and standardize them.
    Xc = X[:, idx_caus]
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)

    # Compute the genetic component of the phenotype.
    yg = Xc.dot(beta)[:, None]
    # Compute the noise component of the phenotype.
    yn = np.sqrt(ve) * rng.standard_normal((X.shape[0], 1))

    # Sum the genetic and noise components to get the simulated phenotype.
    y = yg + yn

    # Initialize the real effect sizes for all variants in X as zeros.
    beta_real = np.zeros(X.shape[1])
    # Update the real effect sizes for the causal variants.
    beta_real[idx_caus] = beta

    # Standardize the phenotypic values to have mean 0 and standard deviation 1.
    ystd = y.std()
    y = (y - y.mean()) / ystd
    # Adjust the real effect sizes accordingly after standardizing y.
    beta_real = beta_real / ystd

    return y, beta_real

def qq_plot(ax, p_values, idx_caus):
    """
    Create a QQ plot given a list of p-values.

    Parameters:
    - p_values: list of p-values
    - figsize: size of the plot
    - title: title for the plot
    """

    # Sort p-values
    p_values = np.array(p_values).flatten()
    idx_caus = np.array(idx_caus).flatten()
    sorted_idx = np.argsort(p_values)
    observed = -np.log10(p_values[sorted_idx])
    expected = -np.log10(np.arange(1, len(p_values) + 1) / (len(p_values) + 2))

    # Create the QQ plot
    plt.scatter(expected, observed, marker='.', label="Non-causal")
    caus_in_sorted = np.argwhere(np.isin(sorted_idx, idx_caus)).flatten()
    plt.scatter(expected[caus_in_sorted], observed[caus_in_sorted], color='red', marker='o', s=20, label='Causal')
    plt.plot([0, max(expected)], [0, max(expected)], color='red', linestyle='--')
    plt.xlabel('Expected -log10(P-value)')
    plt.ylabel('Observed -log10(P-value)')
    plt.legend()



n_causals = [10,20,30,40,50,60,70,80,90,100]
h2s = [0.1,0.2,0.3,0.4,0.5,0.6]

r2_matrix = np.zeros((len(h2s), len(n_causals)))
spearman_matrix = np.zeros((len(h2s), len(n_causals)))
count_v = 0
count_h = 0

indices = np.arange(X_real.shape[0])
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
X_train = X_real[train_idx]
X_test = X_real[test_idx]
train_fam = fam.iloc[train_idx][['fid', 'iid']]
train_fam.to_csv("data/preprocessing/tmp_train_fam.txt", sep=' ', index=False, header=False)
subprocess.run([r"plink_win64_20250819/plink.exe",
                "--bfile", "data/preprocessing/chr22_preprocessed",
                "--keep", r"data/preprocessing/tmp_train_fam.txt",
                "--make-bed",
                "--out", "data/preprocessing/chr22_preprocessed_train_subset",])
bfile = r'data/preprocessing/chr22_preprocessed_train_subset'
bim_train, fam_train, G_train = read_plink(bfile)

pcs = pd.read_csv(r"data/preprocessing/chr22_pca10.eigenvec.zscore", sep=r'\s+', header=None, engine='python')
pcs.columns = ["FID","IID"] + [f"PC{i}" for i in range(1, pcs.shape[1]-1)]
pheno = fam[["fid", "iid"]].copy()
pheno.columns = ["FID", "IID"]

for h2 in h2s:
    for n_c in n_causals:
        idx_caus = rng.choice(X_real.shape[1], size=n_c, replace=False)
        var_expl = np.full(n_c, h2 / n_c)

        y, beta_real = simulate_pheno(X_real, idx_caus, var_expl)

        pheno["y"] = y.reshape(-1)
        df = pheno.merge(pcs, on=["FID", "IID"], how="inner", validate="one_to_one")
        k = 10
        F = np.column_stack([np.ones((df.shape[0], 1)), df[[f"PC{i}" for i in range(1, k+1)]].to_numpy()]).astype(np.float64)

        """
        lmm = LMM(y, F)
        lmm.process(X_real)
        pv = lmm.getPv()
        beta = lmm.getBetaSNP()
        beta_ste = lmm.getBetaSNPste()

        os.makedirs(f"plots/preprocessed_dataset/h2_{h2}/causal_{n_c}", exist_ok=True)

        
        qq_plot(None, pv, idx_caus)
        plt.title(f"QQ plot (n_causals = {n_c} and h2 = {h2})")
        plt.savefig(f"plots/h2_{h2}/causal_{n_c}/qq_h2_{h2}_causal_{n_c}.png")
        plt.close()

        x = bim['pos'].values
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.title(f'Real effect size (n_causals = {n_c} and h2 = {h2})')
        plt.plot(x, beta_real, '.k')
        plt.ylabel('effect size')
        plt.subplot(212)
        plt.title(f'GWAS results (n_causals = {n_c} and h2 = {h2})')
        plt.plot(x, -np.log10(pv), '.k')
        plt.ylabel('-log$_{10}$ P')
        plt.tight_layout()
        plt.savefig(f"plots/h2_{h2}/causal_{n_c}/manhattan_h2_{h2}_causal_{n_c}.png")
        plt.close()
        """

        bim_train_snps = bim_train['snp'].iloc[idx_caus].to_numpy()
        bim_full_snps = bim['snp'].iloc[idx_caus].to_numpy()

        if not np.all(bim_train_snps == bim_full_snps):
            raise Exception(f"Indices did not remain unchanged - invalid run for h2={h2} and n_causal={n_c}")

        y_train = y[train_idx]
        y_test = y[test_idx]

        F_train = F[train_idx]
        lmm = LMM(y_train, F_train)
        lmm.process(X_train)
        pv = lmm.getPv()
        beta = lmm.getBetaSNP()
        beta_ste = lmm.getBetaSNPste()

        """
        os.makedirs(f"plots/train_dataset/h2_{h2}/causal_{n_c}", exist_ok=True)

        qq_plot(None, pv, idx_caus)
        plt.title(f"QQ plot (n_causals = {n_c} and h2 = {h2})")
        plt.savefig(f"plots/train_dataset/h2_{h2}/causal_{n_c}/qq_h2_{h2}_causal_{n_c}.png")
        plt.close()

        x = bim_train['pos'].values
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.title(f'Estimated effect size (n_causals = {n_c} and h2 = {h2})')
        plt.plot(x, beta, '.k')
        plt.ylabel('effect size')
        plt.subplot(212)
        plt.title(f'GWAS results (n_causals = {n_c} and h2 = {h2})')
        plt.plot(x, -np.log10(pv), '.k')
        plt.ylabel('-log$_{10}$ P')
        plt.tight_layout()
        plt.savefig(f"plots/train_dataset/h2_{h2}/causal_{n_c}/manhattan_h2_{h2}_causal_{n_c}.png")
        plt.close()
        """

        gwas_df = pd.DataFrame({
            'CHR': bim_train['chrom'],
            'SNP': bim_train['snp'],
            'BP': bim_train['pos'],
            'P': pv
        })
        os.makedirs("data/tmp_gwas_results", exist_ok=True)
        gwas_df.to_csv(r"data/tmp_gwas_results/train_subset_gwas_results.txt", sep="\t", index=False)

        os.makedirs("data/clumping", exist_ok=True)
        subprocess.run([r"plink_win64_20250819/plink.exe",
                        "--bfile", r"data/preprocessing/chr22_preprocessed_train_subset",
                        "--clump", "data/tmp_gwas_results/train_subset_gwas_results.txt",
                        "--clump-snp-field", "SNP",
                        "--clump-field", "P",
                        "--clump-p1", "1e-4",
                        "--clump-p2", "1e-2",
                        "--clump-r2", "0.07",
                        "--clump-kb", "200",
                        "--out", r"data/clumping/tmp_clumped"])

        clumped = pd.read_csv(
            r"data/clumping/tmp_clumped.clumped", sep=r'\s+', engine='python'
            )

        selected_snps = clumped['SNP'].values

        snp_ids = bim_train['snp'].values  # from full dataset

        # Boolean mask of selected SNPs
        mask = np.isin(snp_ids, selected_snps)

        # Subset training and test genotypes
        X_train_sel = X_train[:, mask]
        X_test_sel = X_test[:, mask]

        # Train linear model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_sel, y_train)

        # Predictions
        y_pred_test = model.predict(X_test_sel)

        # Evaluate
        r2 = r2_score(y_test, y_pred_test)
        rho, _ = spearmanr(y_test, y_pred_test)

        r2_matrix[count_h, count_v] = r2
        spearman_matrix[count_h, count_v] = rho

        count_v += 1
    count_h += 1
    count_v = 0

r2_df = pd.DataFrame(r2_matrix, index=h2s, columns=n_causals)
spearman_df = pd.DataFrame(spearman_matrix, index=h2s, columns=n_causals)

os.makedirs("performances", exist_ok=True)
r2_df.to_csv(r"performances/02split_r2_results.csv")
spearman_df.to_csv(r"performances/02split_spearman_results.csv")

g = sns.heatmap(r2_df, annot=False, cmap="Blues")
g.set(xlabel='Number of causal SNPs', ylabel='Heritability')
g.set_title("R² Score")
plt.savefig("performances/02split_r2_heatmap.png")
plt.close()

g = sns.heatmap(spearman_df, annot=False, cmap="YlOrBr")
g.set(xlabel='Number of causal SNPs', ylabel='Heritability')
g.set_title("Spearman Correlation")
plt.savefig("performances/02split_spearman_heatmap.png")
plt.close()
