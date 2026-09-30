# Public inputs and data access

Raw datasets, image archives, expression matrices and third-party source code are not redistributed here. Obtain inputs from the original providers and retain their license and attribution requirements. MIT applies to this repository's original software, not to third-party datasets or model-generated research facts. Public access alone is not a license grant.

Paths below are relative to the repository root. Create missing directories before placing files. The release's figure rebuild uses supplied results and downloads none of these inputs. Several full-rerun inputs exceed 500 MB; downloading them is a separate, explicit action.

## EU-OPENSCREEN (EUOS)

- [Profile resource v3, Zenodo 19347244](https://zenodo.org/records/19347244): download `Aggregated_Profiles.zip` to `data/Aggregated_Profiles.zip`. Expected size: 1,348,543,344 bytes; SHA-256: `116061a8ee2a719f053989ee54fc64054d7f403df093fd1cb0a7b2994a9432d1`.
- [Annotation archive v1, Zenodo 13309566](https://zenodo.org/records/13309566): download the old `Profile_Analysis.zip` and rename it to `preprocessing/EUOS_feasibility_2026-09-21_v1/Profile_Analysis_v1.zip`. Expected size: 15,394,799 bytes; SHA-256: `930eeb09aa70248dfa824d4f9e6ee0d67ec2748de7af9e8066efc674e7130956`. The newer code ZIP is not a drop-in substitute for these annotation paths.
- The intake script reads archive members, including the corrected U2OS annotation. It does not run the provider's code. The per-member file manifest is included among the frozen reference results.

## JUMP and Tox21

Place these inputs under `preprocessing/JUMP_Tox21_feasibility_2026-09-07_v1/inputs/`:

| Local filename | Source / identity |
| --- | --- |
| `jump_compound.csv.gz` | Compound metadata from [JUMP datasets](https://github.com/jump-cellpainting/datasets); SHA-256 `8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a` |
| `tox21_10k_data_all.sdf.zip` | [NCATS Tox21 challenge training data](https://tripod.nih.gov/tox21/challenge/data.jsp), complete training SDF; 2,627,177 bytes; SHA-256 `024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b` |
| `profiles_var_mad_int_featselect_harmony.parquet` | [Cell Painting Gallery compound profiles](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet); 2,835,051,686 bytes |
| `profiles_var_mad_int (1).parquet` | Count-containing `profiles_var_mad_int.parquet` from the same [Gallery profile location](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int.parquet); retain the local filename shown here; 12,111,419,423 bytes; SHA-256 `42028e8c60692df545e0b1dd087fc9b911f5117c318a8819d768cff251e4edda` |

The profile index used in the project was JUMP datasets v0.11.0. Missing Tox21 labels remain missing rather than being assigned inactive status. JUMP-derived analysis units are also part of the historical identity exclusion set for EUOS.

## EveBio

Obtain files from the [Seal study's EveBio folder](https://github.com/srijitseal/The_Seal_Files/tree/main/The_EveBio_dataset) and place them in `data/positive_controls_2026-09-08/EveBio/`:

- `CP_all_EveBio.csv`: SHA-256 `9d7ea030d03a08f995c8180ce2258b6c115ceb945c7a445e6c5614d22f0e3cf1`.
- `evebio_assay_data_smiles.csv` (from the provider's `data/` subdirectory): SHA-256 `2abd6d2cfb4173222e71bfea74c18f6a12956e0b1ac1c21bcc7e907f478dff48`.

The associated source study is [Seal et al., Nature Communications (2026)](https://doi.org/10.1038/s41467-026-68725-5). Our comparison uses its curated inputs with the splitting and comparator definitions in our code; it is not a reproduction of all source-paper scores.

## Cell Health

Use the [broadinstitute/cell-health repository](https://github.com/broadinstitute/cell-health), release `v2.0`. From `1.generate-profiles/data/consensus/`, place the following in `data/positive_controls_2026-09-08/CellHealth_v2.0/`:

| Filename | SHA-256 |
| --- | --- |
| `cell_painting_modz.tsv.gz` | `b68c0621b957f975b4a82208a2a83ff15ccc884274450beb3310ef4f6c5e4fb4` |
| `cell_health_modz.tsv.gz` | `f4b42f7b41b7cb56800f4691ca79efae874df36408a28c9174bbc4efba15a93e` |
| `cell_painting_median.tsv.gz` | `e90da2f11cc4402b6a015d05578123e847fb94a608ea787b5a916e0d589b2c6c` |
| `cell_health_median.tsv.gz` | `79ad38bc89705ccb2262fd1a8229a8d8292e15dba6fe7fa674a015acc27d9a64` |

Also place these three provider files in the local `author_reference/` subdirectory (keep their basenames):

- `1.generate-profiles/data/labels/feature_mapping_annotated.csv`
- `3.train/results/all_model_predictions_modz.tsv`
- `3.train/results/full_cell_health_regression_modz.tsv.gz`

The last two are required by the original confirmation script's check of the source authors' released predictions. The data comprise 357 guide-by-cell-line profiles from 119 guides targeting 59 genes across three cell lines. These are not 357 independent genes.

## Cardiac transcriptomics and LINCS

- [GEO GSE262419](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262419): place `GSE262419_hash.csv.gz` and `GSE262419_PlateNN.csv.gz` files in `data/raw_public/GSE262419/`.
- [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) and [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138): decompress the corresponding Level 5 GCTX matrices and name them `data/raw_public/GSE92742/GSE92742_L5.gctx` and `data/raw_public/GSE70138/GSE70138_L5.gctx`.
- Place the original `GSE92742_Broad_LINCS_pert_info.txt.gz`, `GSE92742_Broad_LINCS_sig_info.txt.gz`, `GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz` and `GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz` in `data/reference_metadata/`.
- Place `tx4c00193_si_002.xlsx`, the supporting compound table for [DOI 10.1021/acs.chemrestox.4c00193](https://doi.org/10.1021/acs.chemrestox.4c00193), in `data/reference_metadata/`.
- The same directory must contain `KEGG_2021_Human.gmt`, the KEGG 2021 Human library used by the original analysis, available through [Enrichr](https://maayanlab.cloud/Enrichr/#libraries). Respect the provider's terms; do not substitute a newer gene set silently.

The transcriptomic endpoint is a pathway dose-response slope, not directly measured electrical instability or a clinical cardiac outcome. These matrices can be very large and are unnecessary for the quick figure rebuild.

## OASIS

From the [OASIS author repository](https://github.com/broadinstitute/2024_09_09_Axiom_OASIS), retain `2_downstream_analysis/compiled_results/SI_tables/` within the local directory:

`data/positive_controls_2026-09-08/OASIS/repository_snapshot/2024_09_09_Axiom_OASIS-main/`

Only these five source result tables are used: `hit_summary.csv`, `cellpainting_dino_pods.csv`, `mt_pods.csv`, `cellcount_pods.csv`, and `ldh_pods.csv`. The full image collection is not required. The original source project, its code and data retain their own licenses. Our derived complete-case and paired-summary results are supplied under `reference_results/`.

## Version checks

These paths and checksums describe the actual analyzed inputs. A moving upstream branch may no longer contain identical bytes. Resolve a version mismatch before treating a new run as an exact replication; do not suppress intake checks. The repository provides no new dataset DOI, and GitHub hosting is not a substitute for a separately archived release.
