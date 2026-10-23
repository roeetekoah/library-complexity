"""
Q3(d): Reconstruct the pre-PCR fragment-length distribution from the
post-sequencing data.
For each file:
  - per-bin lambda_ell from Q1 single-point estimator
  - reconstructed pre-PCR molecule count: M_hat_ell = U_ell / (1 - exp(-lambda_ell))
    = N_ell / lambda_ell (equal at the MLE)
  - normalised distributions p_obs and p_pre over length bins
Plus per-distribution summary statistics (peak, mean, median, std, IQR)
and a correction-sufficiency check via total variation distance between the
three Sample 1 parts before vs after correction. Output:
q3d_length_distribution.pdf plus summary + TVD tables.
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
MIN_U_BIN = 50

def estimate_lambda(N, U):
    if U == 0 or N == U:
        return np.nan
    return brentq(lambda lam: lam / (1 - np.exp(-lam)) - N / U, 1e-10, N / U)

def bin_data(path):
    df = pd.read_csv(path)
    df["length"] = df["end"] - df["start"] + 1
    df["bin"]    = (df["length"] // BIN_WIDTH) * BIN_WIDTH
    agg = df.groupby("bin").agg(N=("n", "sum"), U=("n", "count")).reset_index()
    agg = agg[agg["U"] >= MIN_U_BIN].copy()
    agg["lam"]     = agg.apply(lambda r: estimate_lambda(r["N"], r["U"]), axis=1)
    agg["bin_mid"] = agg["bin"] + BIN_WIDTH / 2
    agg = agg.dropna(subset=["lam"])

    # M_hat = U / (1 - exp(-lam)) = N / lam (equal at the MLE)
    agg["M_hat"] = agg["N"] / agg["lam"]

    # normalised distributions
    agg["p_pre"] = agg["M_hat"] / agg["M_hat"].sum()   # reconstructed pre-PCR
    agg["p_obs"] = agg["U"]     / agg["U"].sum()        # raw observed (post-PCR)

    return agg

# compute
results = {name: bin_data(path) for name, path in FILES.items()}

# plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {"Sample1_pt1": "steelblue",
          "Sample1_pt2": "cornflowerblue",
          "Sample1_pt3": "lightblue",
          "Sample2":     "tomato"}

for name, df in results.items():
    c = colors[name]
    axes[0].plot(df["bin_mid"], df["p_obs"], color=c, lw=1.2,
                 linestyle="--", label=name)
    axes[1].plot(df["bin_mid"], df["p_pre"], color=c, lw=1.2,
                 label=name)

for ax, title in zip(axes, ["Observed  $U_\\ell / \\Sigma U_{\\ell'}$  (post-PCR)",
                              "Reconstructed  $\\hat{M}_\\ell / \\Sigma \\hat{M}_{\\ell'}$  (pre-PCR)"]):
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Fragment-length distribution: observed vs PCR-bias-corrected", fontsize=11)
plt.tight_layout()
plt.savefig("q3d_length_distribution.pdf", bbox_inches="tight")
plt.show()
print("Figure saved to q3d_length_distribution.pdf")

# distribution statistics
def dist_stats(bin_mid, p):
    """Peak, mean, median, std, IQR of a discrete distribution p over bin_mid."""
    x = np.asarray(bin_mid, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(x)
    xs, ps = x[order], p[order]
    cdf = np.cumsum(ps)
    def q(alpha):
        return float(np.interp(alpha, cdf, xs))
    mean = float(np.sum(ps * xs))
    std  = float(np.sqrt(np.sum(ps * (xs - mean) ** 2)))
    return {
        "peak":   float(xs[np.argmax(ps)]),
        "mean":   mean,
        "median": q(0.5),
        "std":    std,
        "iqr":    q(0.75) - q(0.25),
    }

print(f"\n{'File':<13} {'kind':<5} {'peak':>6} {'mean':>7} {'median':>7} "
      f"{'std':>6} {'IQR':>6}")
print("-" * 56)
for name, df in results.items():
    for kind, col in [("obs", "p_obs"), ("pre", "p_pre")]:
        s = dist_stats(df["bin_mid"], df[col])
        print(f"{name:<13} {kind:<5} {s['peak']:>6.0f} {s['mean']:>7.1f} "
              f"{s['median']:>7.1f} {s['std']:>6.1f} {s['iqr']:>6.1f}")

# correction-sufficiency check via total variation distance
def tvd_on_union(d1, d2, col):
    bins = sorted(set(d1["bin"]).union(d2["bin"]))
    p1 = d1.set_index("bin")[col].reindex(bins, fill_value=0).values
    p2 = d2.set_index("bin")[col].reindex(bins, fill_value=0).values
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return 0.5 * np.abs(p1 - p2).sum()

s1_parts = ["Sample1_pt1", "Sample1_pt2", "Sample1_pt3"]
print("\nTVD between Sample 1 parts (should shrink if correction is working):")
print(f"{'pair':<28} {'TVD(obs)':>10} {'TVD(pre)':>10}  {'change':>9}")
for i, a in enumerate(s1_parts):
    for b in s1_parts[i+1:]:
        tvd_obs = tvd_on_union(results[a], results[b], "p_obs")
        tvd_pre = tvd_on_union(results[a], results[b], "p_pre")
        delta = tvd_pre - tvd_obs
        print(f"{a + ' vs ' + b:<28} {tvd_obs:>10.4f} {tvd_pre:>10.4f}  "
              f"{delta:>+9.4f}")
