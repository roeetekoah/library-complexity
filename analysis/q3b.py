"""
Q3(b): Per-bin likelihood-ratio test for shared lambda across the four files.
For each length bin ell:
  H0: all files share one lambda_ell      (1 free parameter)
  H1: each file has its own lambda_ell^(f) (k free parameters)
LRT statistic = 2 * (ll_H1 - ll_H0); under H0 (Wilks) it is asymptotically
chi^2(k-1). The chi^2 log-survival is computed via scipy.stats.chi2.logsf with
an asymptotic fallback (upper-incomplete-gamma expansion) for chi^2 > 10^4
where scipy itself underflows. BH-FDR adjustment is computed in log10 space
to avoid underflow. Output: q3b_lrt.pdf (lambda curves on top; Manhattan-style
-log10(p) on bottom, log y-axis) + rejection counts at alpha=0.05, BH, and
Bonferroni.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2
from scipy.special import gammaln
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
FILES = {
    "Sample1_pt1": DATA_DIR / "Sample1_pt1.csv",
    "Sample1_pt2": DATA_DIR / "Sample1_pt2.csv",
    "Sample1_pt3": DATA_DIR / "Sample1_pt3.csv",
    "Sample2":     DATA_DIR / "Sample2.csv",
}

BIN_WIDTH = 5
MIN_U_BIN = 50
LN10 = np.log(10)

def estimate_lambda(N, U):
    if U == 0 or N == U:
        return np.nan
    return brentq(lambda lam: lam / (1 - np.exp(-lam)) - N / U, 1e-10, N / U)

def log_likelihood(lam, N, U):
    return -U * lam + N * np.log(lam) - U * np.log(1 - np.exp(-lam))

def chi2_log10_sf(stat, df_):
    """log10 of chi^2 survival, with asymptotic fallback when scipy underflows."""
    val = chi2.logsf(stat, df_)
    if np.isfinite(val):
        return val / LN10
    a = df_ / 2.0
    z = stat / 2.0
    log_sf_approx = (a - 1.0) * np.log(z) - z - gammaln(a)
    return log_sf_approx / LN10

def lrt_shared_lambda(bins_per_file):
    valid = [(N, U) for N, U in bins_per_file if U >= MIN_U_BIN and N > U]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan
    ll_h1 = sum(log_likelihood(estimate_lambda(N, U), N, U) for N, U in valid)
    N_pool = sum(N for N, U in valid)
    U_pool = sum(U for N, U in valid)
    lam_pool = estimate_lambda(N_pool, U_pool)
    ll_h0 = log_likelihood(lam_pool, N_pool, U_pool)
    stat = 2 * (ll_h1 - ll_h0)
    df_ = len(valid) - 1
    log10_p = chi2_log10_sf(stat, df_)
    return stat, df_, log10_p

def bin_data(path):
    df = pd.read_csv(path)
    df["length"] = df["end"] - df["start"] + 1
    df["bin"] = (df["length"] // BIN_WIDTH) * BIN_WIDTH
    agg = df.groupby("bin").agg(N=("n", "sum"), U=("n", "count")).reset_index()
    agg = agg[agg["U"] >= MIN_U_BIN].copy()
    agg["lam"] = agg.apply(lambda r: estimate_lambda(r["N"], r["U"]), axis=1)
    agg["bin_mid"] = agg["bin"] + BIN_WIDTH / 2
    return agg.dropna(subset=["lam"])

binned = {name: bin_data(path) for name, path in FILES.items()}

all_bins = sorted(set().union(*[set(b["bin"]) for b in binned.values()]))
lrt_rows = []
for b in all_bins:
    pairs = []
    for name, df in binned.items():
        row = df[df["bin"] == b]
        if len(row):
            pairs.append((int(row["N"].iloc[0]), int(row["U"].iloc[0])))
        else:
            pairs.append(None)
    valid_pairs = [p for p in pairs if p is not None]
    stat, df_lrt, log10_p = lrt_shared_lambda(valid_pairs)
    lrt_rows.append({"bin_mid": b + BIN_WIDTH / 2, "lrt": stat, "df": df_lrt, "log10p": log10_p})

lrt_df = pd.DataFrame(lrt_rows).dropna()

def bh_fdr_log10(log10_pvals):
    n = len(log10_pvals)
    log10_p = np.asarray(log10_pvals, dtype=float)
    order = np.argsort(log10_p)
    log10_sorted = log10_p[order]
    log10_q = log10_sorted + np.log10(n / np.arange(1, n + 1))
    log10_q = np.minimum.accumulate(log10_q[::-1])[::-1]
    log10_q = np.minimum(log10_q, 0.0)
    out = np.empty(n)
    out[order] = log10_q
    return out

n_tests = len(lrt_df)
lrt_df["log10q"] = bh_fdr_log10(lrt_df["log10p"].values)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                gridspec_kw={"height_ratios": [2.4, 1.4]})

for name, df in binned.items():
    ax1.plot(df["bin_mid"], df["lam"], ".-", ms=3, lw=1, label=name)
ax1.set_ylabel(r"$\hat{\lambda}$")
ax1.set_title(r"$\hat{\lambda}$ vs fragment length with per-bin LRT significance")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Bottom panel: log y-axis so reference lines at 1.3 and 3.26 are visible
# against the 1286 max. Floor data at 0.05 so log-scale shows everything.
neg_log10p = np.maximum(-lrt_df["log10p"].values, 0.05)
neg_log10q = np.maximum(-lrt_df["log10q"].values, 0.05)
ax2.semilogy(lrt_df["bin_mid"], neg_log10p, "k.-", ms=3, lw=1,
             label=r"$-\log_{10}$(raw $p$)")
ax2.semilogy(lrt_df["bin_mid"], neg_log10q, "r.-", ms=3, lw=1,
             label=r"$-\log_{10}$(BH $q$)")
ax2.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1.2,
            label=r"$\alpha=0.05$")
ax2.axhline(-np.log10(0.05 / n_tests), color="purple", linestyle=":", linewidth=1.2,
            label=f"Bonferroni $\\alpha/{n_tests}$")
ax2.set_xlabel("Fragment length (bp)")
ax2.set_ylabel(r"$-\log_{10}(p)$")
ax2.legend(fontsize=7, loc="lower right")
ax2.grid(True, which="both", alpha=0.3)
ax2.set_ylim(0.1, 3000)

plt.tight_layout()
plt.savefig("q3b_lrt.pdf", bbox_inches="tight")
plt.show()
print("Figure saved to q3b_lrt.pdf")

n_sig_raw  = (lrt_df["log10p"] < np.log10(0.05)).sum()
n_sig_fdr  = (lrt_df["log10q"] < np.log10(0.05)).sum()
n_sig_bonf = (lrt_df["log10p"] < np.log10(0.05 / n_tests)).sum()
print(f"\nBins tested: {n_tests}")
print(f"Significant (raw p<0.05):    {n_sig_raw}  ({100*n_sig_raw/n_tests:.0f}%)")
print(f"Significant (FDR q<0.05):    {n_sig_fdr} ({100*n_sig_fdr/n_tests:.0f}%)")
print(f"Significant (Bonferroni p<{0.05/n_tests:.2e}):  {n_sig_bonf} ({100*n_sig_bonf/n_tests:.0f}%)")
print(f"-log10(p) range: [{neg_log10p.min():.1f}, {neg_log10p.max():.1f}]")
