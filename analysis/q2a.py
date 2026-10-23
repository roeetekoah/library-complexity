"""
Q2(a): Estimate lambda on each file using
  Method 1: single-point MLE (brentq root-find on lambda/(1-exp(-lambda)) = N/U)
  Method 2: subsampling curve fit
            - MC subsampling via per-fragment hypergeometric (5 trials per depth)
            - Analytical rarefaction (Heck/Hurlbert) via gammaln in log space
            - Fit M by minimizing SSE against E[U_obs] = M*(1 - exp(-N_sub/M))
Outputs q2a_saturation.pdf (2x2 panel, one file per panel) and a comparison table.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
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

DEPTHS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
N_MC_TRIALS = 5  # Monte Carlo replicates per depth fraction
RNG = np.random.default_rng(0)

# ── estimators ────────────────────────────────────────────────────────────────

def single_point_lambda(N, U):
    """Single-point root-find: lambda/(1-exp(-lambda)) = N/U  (Setup / Q1 MLE)."""
    return brentq(lambda lam: lam / (1 - np.exp(-lam)) - N / U, 1e-10, N / U)

def rarefaction_exact(nj, N, m):
    """
    Analytical rarefaction (Heck / Hurlbert):
        E[U_obs(m)] = sum_j [1 - C(N - n_j, m) / C(N, m)]
    Computed in log space via gammaln to avoid overflow at N ~ 1e6.
    """
    # log C(a, b) = gammaln(a+1) - gammaln(b+1) - gammaln(a-b+1)
    log_num = gammaln(N - nj + 1) - gammaln(m + 1) - gammaln(N - nj - m + 1)
    log_den = gammaln(N + 1)     - gammaln(m + 1) - gammaln(N - m + 1)
    p_miss = np.exp(log_num - log_den)
    p_miss = np.where(N - nj < m, 0.0, p_miss)   # if can't avoid, p_miss = 0
    return float(np.sum(1.0 - p_miss))

def mc_one_trial(nj, m):
    """
    Monte Carlo subsampling: per-fragment Hypergeometric(N, nj, m) draws.
    Marginals are exact; joint sum is relaxed (counts independent), but
    E[U_obs] depends only on marginals, so the estimator is unbiased.
    """
    counts = RNG.hypergeometric(ngood=nj, nbad=int(nj.sum()) - nj, nsample=m)
    return int(np.sum(counts > 0))

def fit_M_to_curve(Nsubs, Uobs):
    """Fit M by minimising SSE between observed Uobs and E[U] = M*(1 - exp(-N/M))."""
    def sse(M):
        pred = M * (1.0 - np.exp(-Nsubs / M))
        return float(np.sum((Uobs - pred) ** 2))
    res = minimize_scalar(sse, bounds=(max(Uobs), 50 * max(Uobs)), method="bounded")
    return float(res.x)

# ── per-file pipeline ─────────────────────────────────────────────────────────

def process_file(name, path):
    df = pd.read_csv(path)
    nj = df["n"].values.astype(np.int64)
    U_full = len(nj)
    N_full = int(nj.sum())

    # Method 1: single-point MLE
    lam_sp = single_point_lambda(N_full, U_full)
    M_sp   = N_full / lam_sp

    # Method 2a: analytical rarefaction (Heck) at each depth fraction
    N_sub = (DEPTHS * N_full).astype(np.int64)
    U_heck = np.array([rarefaction_exact(nj, N_full, m) for m in N_sub])

    # Method 2b: MC subsampling at each depth, mean +/- sd over trials
    mc_means, mc_sds = [], []
    for m in N_sub:
        trials = [mc_one_trial(nj, m) for _ in range(N_MC_TRIALS)]
        mc_means.append(np.mean(trials))
        mc_sds.append(np.std(trials, ddof=1) if N_MC_TRIALS > 1 else 0.0)
    U_mc_mean = np.array(mc_means)
    U_mc_sd   = np.array(mc_sds)

    # Fit M to the Heck curve (the MC mean should give the same answer +/- noise)
    M_fit_heck = fit_M_to_curve(N_sub, U_heck)
    M_fit_mc   = fit_M_to_curve(N_sub, U_mc_mean)

    lam_fit = N_full / M_fit_heck

    return {
        "name": name,
        "N": N_full, "U": U_full,
        "lam_sp": lam_sp, "M_sp": M_sp,
        "lam_fit": lam_fit,
        "M_fit_heck": M_fit_heck, "M_fit_mc": M_fit_mc,
        "N_sub": N_sub,
        "U_heck": U_heck,
        "U_mc_mean": U_mc_mean, "U_mc_sd": U_mc_sd,
    }

results = {name: process_file(name, path) for name, path in FILES.items()}

# ── plot 2x2 saturation curves ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.suptitle("Saturation curves: single-point vs Monte Carlo vs analytical rarefaction",
             fontsize=12)

for ax, (name, r) in zip(axes.flatten(), results.items()):
    # MC: filled red squares with errorbars; MC and Heck are essentially equal
    # (Heck is the exact expectation of MC), so we plot MC first and put a
    # blue open-circle outline of Heck on top so both are visible.
    # Errorbars use a wider amplification so the tiny std is at least visible.
    err_amp = 50.0  # visual amplification — std is ~1/sqrt(5) of single-trial std and dwarfed by U
    ax.errorbar(r["N_sub"], r["U_mc_mean"], yerr=r["U_mc_sd"] * err_amp,
                fmt="rs", ms=6, mfc="red", mec="darkred", capsize=4,
                elinewidth=1.2, ecolor="darkred",
                label=f"Monte Carlo (mean ± {err_amp:.0f}×sd)")
    # Heck analytical -- open blue circles ON TOP of MC so they ring the red square
    ax.plot(r["N_sub"], r["U_heck"], "o", ms=11, mfc="none", mec="blue",
            mew=1.6, label="Analytical (Heck)")
    # Single-point fit curve
    Ngrid = np.linspace(0, r["N"], 200)
    ax.plot(Ngrid, r["M_fit_heck"] * (1 - np.exp(-Ngrid / r["M_fit_heck"])),
            "g:", lw=1.2, label=f"Fit M = {r['M_fit_heck']:.3e}")
    ax.axhline(r["M_fit_heck"], color="k", lw=0.5, alpha=0.5)
    ax.set_title(name)
    ax.set_xlabel("$N_{sub}$ (reads)")
    ax.set_ylabel("$U_{obs}$ (unique fragments)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("q2a_saturation.pdf", bbox_inches="tight")
plt.show()
print("Figure saved to q2a_saturation.pdf")

# ── summary table ─────────────────────────────────────────────────────────────

print(f"\n{'File':<14} {'lam_sp':>8} {'lam_fit':>8} {'M_sp':>14} "
      f"{'M_fit_heck':>14} {'M_fit_mc':>14} {'M_sp/M_fit':>12}")
print("-" * 100)
for name, r in results.items():
    ratio = r["M_sp"] / r["M_fit_heck"]
    print(f"{name:<14} {r['lam_sp']:>8.3f} {r['lam_fit']:>8.3f} "
          f"{r['M_sp']:>14.3e} {r['M_fit_heck']:>14.3e} "
          f"{r['M_fit_mc']:>14.3e} {ratio:>12.4f}")
