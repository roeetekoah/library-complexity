"""
Q3(a): Per-bin lambda estimation as a function of fragment length.
For each file:
  - bin reads into 5 bp length bins, ell = end - start + 1
  - for each bin, compute (N_ell, U_ell) and estimate lambda_ell by the
    single-point root-find from Q1 (brentq on lam/(1 - exp(-lam)) = N/U)
  - discard bins with U_ell < 50 (too noisy)
Output: q3a_lambda_vs_length.pdf and per-file summary table
(peak lambda, peak location, mean lambda, number of retained bins).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
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
MIN_U_BIN = 50   # discard bins with too few fragments for reliable estimation

def estimate_lambda(N, U):
    if U == 0 or N == U:   # all singletons → λ→0; skip
        return np.nan
    ratio = N / U
    return brentq(lambda lam: lam / (1 - np.exp(-lam)) - ratio, 1e-10, ratio)

def lambda_by_length(df, bin_width=BIN_WIDTH, min_u=MIN_U_BIN):
    df = df.copy()
    df["length"] = df["end"] - df["start"] + 1
    df["bin"] = (df["length"] // bin_width) * bin_width  # left edge of bin

    results = []
    for bin_left, group in df.groupby("bin"):
        N_bin = group["n"].sum()
        U_bin = len(group)
        if U_bin < min_u:
            continue
        lam = estimate_lambda(N_bin, U_bin)
        if not np.isnan(lam):
            results.append({
                "bin_mid": bin_left + bin_width / 2,
                "lambda":  lam,
                "U":       U_bin,
                "N":       N_bin,
            })
    return pd.DataFrame(results)

# ── compute ────────────────────────────────────────────────────────────────────

results = {}
for name, path in FILES.items():
    df = pd.read_csv(path)
    results[name] = lambda_by_length(df)
    print(f"{name}: {len(results[name])} bins, "
          f"length range [{results[name]['bin_mid'].min():.0f}, "
          f"{results[name]['bin_mid'].max():.0f}] bp")

# ── plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

for name, res in results.items():
    ax.plot(res["bin_mid"], res["lambda"], ".-", ms=3, lw=1, label=name)

ax.set_xlabel("Fragment length (bp)")
ax.set_ylabel(r"$\hat{\lambda}$ (reads per molecule)")
ax.set_title(r"$\hat{\lambda}$ vs fragment length  (5 bp bins, single-point estimator)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("q3a_lambda_vs_length.pdf", bbox_inches="tight")
plt.show()
print("Figure saved to q3a_lambda_vs_length.pdf")
