# Poverty and ethnic diversity predict US metropolitan murder rates over 60 years

Reproducibility materials for the manuscript submitted to *Nature Cities*. The study analyzes murder, demographic, and socioeconomic measures for 344 US metropolitan statistical areas from 1960 through 2020.

The submitted manuscript is included in [`manuscript/`](manuscript/). The cover letter is not included.

## Quick start

The repository contains the analysis code and environment used for the manuscript. Processed inputs are available to editors and reviewers upon request but are not distributed publicly pending confirmation of the source providers' redistribution terms.

```bash
conda env create -f environment.yml
conda activate nature-cities-murders
jupyter lab
```

Open the notebooks from the repository root and follow the order in [`notebooks/README.md`](notebooks/README.md). Once the processed analysis files have been placed under `data/`, notebook 07 is the shortest route to Figures 1–2.

## Repository map

| Path | Contents |
|---|---|
| `manuscript/` | Submitted manuscript and bibliography |
| `notebooks/` | Data preparation and figure notebooks |
| `src/` | Plotting, statistics, and PyStan functions |
| `data/` | Source notes and expected local data layout |
| `results/cross_sectional/` | Saved cross-sectional scaling estimates |
| `figures/` | Manuscript figure outputs |
| `environment.yml` | Python 3.9 / PyStan 2 environment |
| `REPRODUCIBILITY_STATUS.md` | Figure-to-code map and known gaps |

## Analysis order

1. Notebooks 01–04 document raw FBI and Census/NHGIS data preparation.
2. Notebooks 05–06 construct the processed MSA analysis table.
3. Notebook 07 generates Figures 1–2.
4. Notebook 08 contains the Bayesian workflow for Figures 3–4.
5. Notebook S1 generates Supplementary Figure S1.
6. Notebook 09 generates Supplementary Figure S2.

The source and processed data are not redistributed here. Editors and reviewers may request the processed files from the corresponding author. See [`data/README.md`](data/README.md) for sources and expected file locations.

## Bayesian output files

The fitted Bayesian posterior files used for Figures 3, 4, and S2 are not stored in GitHub. Their expected paths are listed in [`QUEST_RESULTS.md`](QUEST_RESULTS.md). The models can be refitted after obtaining the processed data, but this takes substantially longer than the non-Bayesian notebooks.

The Bayesian helper is configured to match the final manuscript: 7,000 total iterations per chain, including 6,000 warm-up iterations and 1,000 retained iterations, using two chains and no thinning. See [`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md).

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Original software and documentation are released under the [`MIT License`](LICENSE). That license does not grant rights to third-party datasets; the source-specific access and reuse conditions described in [`data/README.md`](data/README.md) continue to apply.

