# Reproducibility notes

The analysis code is included. Processed data, large fitted Bayesian objects, and original provider files are not stored in GitHub.

## Manuscript-to-code map

| Manuscript output | Source | Included inputs | Current status |
|---|---|---:|---|
| Figure 1A | `notebooks/07_figures_1_2_local.ipynb` | On request | Code and final PDF included |
| Figure 1B | `notebooks/07_figures_1_2_local.ipynb` | On request | Code and final PDF included |
| Figure 2A | `notebooks/07_figures_1_2_local.ipynb` | On request | Code and final PDF included |
| Figure 2B–D | `notebooks/07_figures_1_2_local.ipynb` | On request | Code and final PDF included |
| Figure 3A–D | `notebooks/08_figures_3_4_bayesian.ipynb` | Processed data on request; posterior files not included | Code and reference PDFs included |
| Figure 4 | `notebooks/08_figures_3_4_bayesian.ipynb` | Processed data on request; posterior files not included | Code and reference PDF included |
| Figure S1 | `notebooks/S1.ipynb` | On request | Code and final PDF included |
| Figure S2 | `notebooks/09_supplementary_urban_scaling.ipynb` | Processed data on request; posterior files not included | Code included; standalone final PDF unavailable |

## Checks performed

- The initial-submission manuscript is a readable 23-page PDF.
- The analysis table has 344 MSAs covering 1960–2020, with no duplicate MSA-year rows or missing values.
- Notebooks 05, 06, 07, and S1 were run from the processed inputs during preparation of this repository.
- A Python 3.9 / PyStan 2.19.1.1 environment is included.
- The Bayesian helper matches the manuscript sampling configuration: 7,000 total iterations, 6,000 warm-up iterations, 1,000 retained iterations, two chains, and no thinning.

The current workflow corrects a historical `MY`/`WY` state-code typo and includes Casper and Cheyenne, giving the 344 MSAs reported in the manuscript.

## Files not included

1. The 59 fitted Bayesian posterior files listed in `QUEST_RESULTS.md`.
2. Raw FBI, ICPSR, Census/NHGIS, CDC, and BLS files. Access notes are in `data/README.md`.


