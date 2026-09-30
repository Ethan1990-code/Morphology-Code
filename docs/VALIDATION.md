# Release validation

Checked on 2026-09-30, using Python 3.13.13 on Windows and the versions in `requirements.txt`.

## Checks performed

- All 25 original analysis, preprocessing and figure scripts were compared with their project sources. Changes were limited to portable input/output paths and repository-root locations; statistical calculations, splits, seeds, model grids and plotting logic were retained.
- SHA-256 integrity was checked for all 56 source-manifest entries: 25 scripts, 2 original specifications and 29 result/source-manifest CSVs.
- The 8 tests in `tests/test_release.py` passed. They cover source integrity, syntax, the existing train-only preprocessing and similarity self-test, labeled-compound partition counts and scaffold separation, recomputation of archived Brier/AP/AUROC metrics, one archived-seed example from each of 5 simulation scenarios, all 22 table-source mappings, and refusal to overwrite different existing results.
- The figure builder completed and produced 9 figures and 21 registered panels. All 22 panel/source-manifest CSV outputs matched the manuscript figure source tables as parsed tables. The rendered overview was visually inspected for clipping and obvious layout failures.
- Image files were not pixel-identical to the prior rendered TIFFs despite matching dimensions. Pixel-level rendering is not claimed reproducible across environments; numerical panel-source equality was verified. The original submission images were not replaced.
- The publication file list was scanned for local user/machine paths, common credential patterns and private-key markers. No matching sensitive patterns were found. Raw input datasets, manuscript files and internal work logs are excluded from the release.

## Not performed in release packaging

- No complete raw-data training rerun, re-extraction of source image features or new large download.
- No independent scientific peer review and no new validation of the paper's generalizability.
- No fresh dependency installation or cross-platform certification. Exact font and raster appearance can differ.
- No claim that a GitHub repository already has a DOI or a separately archived immutable release.

The frozen results enable inspection and figure rebuilding. They must not be confused with outputs newly produced by full model training. Use the reproduction guide to distinguish the two paths.
