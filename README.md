# Morphology-Code

Analysis and figure-generation code for **Context-dependent gains from cell morphology**.

This study asks when cell morphology adds predictive information beyond chemical structure and cell-count-derived features, and whether that gain persists across measurement contexts. The main EUOS analysis is accompanied by JUMP, EveBio, Cell Health, cardiac transcriptomic and OASIS comparisons and known-answer simulations. The repository preserves unfavorable as well as favorable comparisons; it does not implement a universally validated safety or reliability framework.

## Start here

Use Python 3.13 and an isolated environment. Install the tested dependency versions:

```text
git clone https://github.com/Ethan1990-code/Morphology-Code.git
cd Morphology-Code
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell), then:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To rebuild the paper's 4 main and 5 supplementary figures without large data downloads:

```text
python tools/restore_reference_results.py
python figures/build_figures_v81.py
```

TIFF (600 dpi), PDF and SVG outputs appear in `outputs/Figures/`. Panel-level source data and PNG previews appear in `outputs/figure_build/`. This path replots supplied results; it does **not** rerun model training. Use a separate fresh clone for a raw-data rerun, following the [reproduction guide](docs/REPRODUCING.md).

## Repository contents

| Directory | Purpose |
| --- | --- |
| `analysis/` | Final analyses and the earlier scripts on which they depend |
| `preprocessing/` | Public-data identity matching, profile preparation and count-proxy extraction |
| `figures/` | Final manuscript figure builder |
| `reference_results/` | Frozen result CSVs used for figures and tables, including compound-level predictions; no patient data |
| `docs/` | Data access, execution order, validation scope and mapping for Tables 1-2 and S1-S20 |
| `provenance/` | SHA-256 manifest connecting published files to their source versions |
| `tests/`, `tools/` | Bounded release checks and a non-overwriting result-restore helper |

See [data sources](docs/DATA_SOURCES.md), [table-to-result mapping](docs/table_source_map.json) and [validation record](docs/VALIDATION.md). Raw data, large archives, Word manuscripts and private working files are excluded. Dated directory names preserve the provenance and dependencies of the published analyses.

## Important interpretation and reproducibility limits

- Use **EUOS v2**, the final corrected analysis. A compound-identity/scaffold representation issue was corrected after v1 scores had been seen. This is a disclosed corrected reanalysis, not another independent blinded experiment.
- Only two EUOS endpoints qualified for the primary external evaluation. Supporting datasets address related but different questions; they are not interchangeable external replications.
- The adapted similarity comparator follows a documented formula slice, not a complete reproduction of the source publication's benchmark.
- The release was prepared by relocating paths, retaining statistical code and checking frozen outputs. A new complete raw-data model-fitting run was not performed for packaging. Package pins describe the tested packaging environment, not a claim that every historical run used identical versions.

## License and citation

Original software is released under the [MIT License](LICENSE). Upstream data and third-party software remain subject to their own licenses and attribution requirements; this repository does not relicense them. Consult the linked sources before redistribution.

For reproducible citation, use this repository URL and the exact commit used. No software DOI or paper publication DOI is asserted by this release. Please report reproducibility issues through the repository's issue tracker, including the commit, environment, command and error message, without uploading confidential data.
