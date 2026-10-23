"""
Q2(b): Pearson chi-squared goodness-of-fit on per-fragment count histograms.
For each file:
  - estimate lambda by single-point (Q1 MLE) and curve-fit (Q2a Method 2)
  - build observed histogram obs[k] = # fragments with n_j = k for k = 1, 2, ...
  - compute expected histogram exp[k] = U * P_obs(k; lambda_hat)
  - Cochran tail-merging: collapse right-tail bins until every retained bin
    has exp >= 5
  - Pearson chi^2 = sum (obs - exp)^2 / exp
  - df = K - 1 - 1 (K bins after merging, -1 for total constraint, -1 for
    lambda estimated by MLE -- the Wilks correction)
Output: q2b_histograms.pdf (4 panels, log y-axis) and a comparison table.
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

COCHRAN_MIN = 5.0
DEPTHS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

def estimate_lambda_single(N, U):
    return brentq(lambda lam: lam / (1 - np.exp(-lam)) - N / U, 1e-10, N / U)

def rarefaction_exact(nj, N, m):
    log_num = gammaln(N - nj + 1) - gammaln(m + 1) - gammaln(N - nj - m + 1)
    log_den = gammaln(N + 1) - gammaln(m + 1) - gammaln(N - m + 1)
    p_miss = np.exp(log_num - log_den)
    p_miss = np.where(N - nj < m, 0.0, p_miss)
    return float(np.sum(1.0 - p_miss))

def estimate_lambda_curve(nj, N, U):
    N_sub = (DEPTHS * N).astype(np.int64)
    U_heck = np.array([rarefaction_exact(nj, N, m) for m in N_sub])
    def sse(M):
        pred = M * (1.0 - np.exp(-N_sub / M))
        return float(np.sum((U_heck - pred) ** 2))
    res = minimize_scalar(sse, bounds=(max(U_heck), 50 * max(U_heck)), method="bounded")
    return N / float(res.x)

def trunc_poisson_pmf(k, lam):
    log_pmf = -lam + k * np.log(lam) - gammaln(k + 1) - np.log(1 - np.exp(-lam))
    return np.exp(log_pmf)

def pearson_chi2(nj, lam):
    U = len(nj)
    k_max = int(max(nj))
    k_grid = np.arange(1, k_max + 1)
    obs = np.array([np.sum(nj == k) for k in k_grid], dtype=float)
    exp = U * trunc_poisson_pmf(k_grid, lam)
    # Cochran tail-merge from the right
    i = len(exp) - 1
    while i > 0 and exp[i] < COCHRAN_MIN:
        exp[i - 1] += exp[i]
        obs[i - 1] += obs[i]
        exp = exp[:i]; obs = obs[:i]; k_grid = k_grid[:i]
        i -= 1
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    dof = len(exp) - 1 - 1  # -1 total, -1 lambda estimated (Wilks)
    return chi2, dof

# ── plot 4 panels and print table ────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
plt.suptitle(r"Histogram of $n_j$ vs truncated-Poisson prediction", fontsize=12)

print(f"{'File':<15} {'lam_sp':>10} {'lam_curve':>11} {'X2_sp':>10} {'X2_curve':>10} {'df':>5}")
print("-" * 65)

for ax, (name, path) in zip(axes, FILES.items()):
    df_data = pd.read_csv(path)
    nj = df_data["n"].values
    N, U = int(nj.sum()), len(nj)
    lam_sp = estimate_lambda_single(N, U)
    lam_curve = estimate_lambda_curve(nj, N, U)
    chi2_sp, dof_sp = pearson_chi2(nj, lam_sp)
    chi2_curve, _ = pearson_chi2(nj, lam_curve)
    print(f"{name:<15} {lam_sp:>10.4f} {lam_curve:>11.4f} "
          f"{chi2_sp:>10.0f} {chi2_curve:>10.0f} {dof_sp:>5d}")
    k_full = np.arange(1, int(nj.max()) + 1)
    obs_full = np.array([np.sum(nj == kv) for kv in k_full], dtype=float)
    pred_sp = U * trunc_poisson_pmf(k_full, lam_sp)
    pred_curve = U * trunc_poisson_pmf(k_full, lam_curve)
    ax.bar(k_full, obs_full, width=0.7, color="steelblue", alpha=0.7, label="Observed")
    ax.plot(k_full, pred_sp, "r.-", lw=1, ms=4,
            label=f"Single-point $\\hat\\lambda$ = {lam_sp:.3f}")
    ax.plot(k_full, pred_curve, "g.-", lw=1, ms=4,
            label=f"Curve-fit $\\hat\\lambda$ = {lam_curve:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("$n_j$ (reads per molecule)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{name}  |  $\\chi^2$: sp={chi2_sp:.0f}, curve={chi2_curve:.0f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("q2b_histograms.pdf", bbox_inches="tight")
plt.show()
print("\nFigure saved to q2b_histograms.pdf")
