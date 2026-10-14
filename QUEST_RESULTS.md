# Files to copy from Northwestern Quest

The workflow uses 59 fitted posterior files. The complete, copy-ready path list is in [`quest_download_manifest.txt`](quest_download_manifest.txt).

## Urban-scaling models (22 files)

Copy `Murder.pkl` for every model from `model_1` through `model_22`:

```text
results/5000/results/urban_scaling/model_1/Murder.pkl
results/5000/results/urban_scaling/model_2/Murder.pkl
...
results/5000/results/urban_scaling/model_22/Murder.pkl
```

These files support the urban-scaling model comparison and Figure S2 in the candidate local Bayesian notebook.

## Cobb–Douglas models (37 files)

Copy `Murder.pkl` for every model from `model_1` through `model_37`:

```text
results/5000/results/cobb_douglas/model_1/Murder.pkl
results/5000/results/cobb_douglas/model_2/Murder.pkl
...
results/5000/results/cobb_douglas/model_37/Murder.pkl
```

These files support the model- and feature-selection comparisons and Figures 3 and 4. In particular, the displayed coefficient and prediction plots directly use model 23; later exploratory cells also access models 31 and 32.

## Transfer example

Run from the local repository root, replacing the placeholder with the actual Quest login and project path:

```bash
rsync -av --progress \
  NETID@login.quest.northwestern.edu:/path/to/project/results/5000/results/ \
  results/5000/results/
```

Then validate the transfer:

```bash
python scripts/check_quest_results.py
```

If you must download files individually, use every line in `quest_download_manifest.txt` as the relative Quest source path and preserve that directory structure locally.

The posterior directory is intentionally ignored by Git because the pickle files may be large. For publication, archive them on Zenodo, Dryad, or OSF and record the DOI and checksums in this repository.

Only load pickle files produced by your own trusted Quest jobs; pickle files can execute code when opened.
