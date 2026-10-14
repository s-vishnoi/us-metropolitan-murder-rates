# Notebook order

Run the main analysis notebooks in numerical order:

The data files are not distributed through GitHub. Their expected paths and sources are listed in `data/README.md`; processed files are available to editors and reviewers upon request.

1. `01_parsing_crime_data.ipynb`
2. `02_matching_crime_data.ipynb`
3. `03_fix_crime_missing_reports.ipynb`
4. `04_interpolate_demographic_and_socioeconomic_data-2.ipynb`
5. `05_merge_demographic_and_socioeconomics_metrics.ipynb`
6. `06_merge_datasets.ipynb`
7. `07_figures_1_2_local.ipynb` — local-system analysis and Figures 1–2

`S1.ipynb` contains supplementary analysis and is not a prerequisite for notebook 07.

The Bayesian workflow is split into two reproducible entry points:

- `08_figures_3_4_bayesian.ipynb` — Figures 3–4; requires the Cobb--Douglas posterior files.
- `09_supplementary_urban_scaling.ipynb` — notebook entry point for Figure S2; requires the urban-scaling posterior files.
- `09_supplementary_urban_scaling.py` — the same Figure S2 workflow as a non-interactive script.

The local Figure 1–2 notebook automatically finds the repository root, changes its working directory to `notebooks/` so historical relative data paths remain valid, and adds the repository root to Python's import path.

Local experimental notebooks and duplicate helper copies are retained outside version control for provenance; they predate or duplicate the canonical workflow listed above.
