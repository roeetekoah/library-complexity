# Running the analysis

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install -r requirements.txt
python tools/demo.py
```

The demonstration generates four small synthetic tables in a temporary directory, invokes all five unmodified analysis scripts with Matplotlib's noninteractive backend, and copies five PDF plots and a log to `demo-output/`. It does not require or download sequencing data. On failure it exits with a nonzero status. The synthetic plots retain the original script titles; the output directory's README and run log identify them as synthetic. They must not be used as biological results.

## Original-data route

The source coursework used four supplied CSVs: `Sample1_pt1.csv`, `Sample1_pt2.csv`, `Sample1_pt3.csv`, and `Sample2.csv`. They are not included here; their redistribution terms are unverified. If you have authorized access, place them in a `data/` directory under your chosen working directory. Required columns are `chr`, `start`, `end`, and positive integer `n`; fragment length is inclusive, `end - start + 1`. Each row represents a distinct observed fragment, not one read.

Run, for example, `python analysis/q2a.py` from the repository root. The five scripts write PDFs to the current working directory and call `plt.show()`; set the `MPLBACKEND` environment variable to `Agg` for unattended runs. The original multi-million-row tables require appreciably more memory and time than the demonstration.

| Script | Output |
|---|---|
| `q2a.py` | `q2a_saturation.pdf` and estimator table |
| `q2b.py` | `q2b_histograms.pdf` and goodness-of-fit table |
| `q3a.py` | `q3a_lambda_vs_length.pdf` |
| `q3b.py` | `q3b_lrt.pdf` and rejection counts |
| `q3d.py` | `q3d_length_distribution.pdf`, summaries and TVD comparisons |

## Documents

The original report and background primers are provided as PDFs with accompanying TeX source. Report and primer figures sit alongside the corresponding source. Compile each document from its own directory with a standard LaTeX installation; a TeX runtime is not required for the Python demonstration.

The demo verifies that the scripts execute end to end on a small valid fixture. It is not a statistical validation suite or a fresh reproduction of the report's numerical tables. See [the methods guide](METHODS.md) for assumptions and interpretation limits.
