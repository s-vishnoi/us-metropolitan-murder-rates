"""Generate Supplementary Figure S2 from archived urban-scaling posteriors.

Run from the repository root after restoring the 22 trusted posterior files under
results/5000/results/urban_scaling/model_<n>/Murder.pkl.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import pystan_cobb_douglas_mod as cb


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "5000" / "results" / "urban_scaling"
OUTPUT = ROOT / "figures" / "sm" / "urban_model_comparison_waic.pdf"

MODEL_NAMES = {
    "model_1": "Pooled",
    "model_2": "Time-varying intercepts",
    "model_3": "Time-varying slopes",
    "model_4": "Time-varying intercepts\nand slopes",
    "model_5": "Region-varying intercepts",
    "model_6": "Region-varying slopes",
    "model_7": "Region-varying intercepts\nand slopes",
    "model_8": "Time- and region-varying intercepts",
    "model_9": "Time- and region-varying intercepts\nand time-varying slopes",
    "model_10": "Time- and region-varying\nintercepts and slopes",
    "model_11": "Random walk priors",
    "model_12": "Pooled TL",
    "model_13": "Time-varying intercepts TL",
    "model_14": "Time-varying slopes TL",
    "model_15": "Time-varying intercepts\nand slopes TL",
    "model_16": "Region-varying intercepts TL",
    "model_17": "Region-varying slopes TL",
    "model_18": "Region-varying intercepts\nand slopes TL",
    "model_19": "Time- and region-varying intercepts TL",
    "model_20": "Time- and region-varying intercepts\nand time-varying slopes TL",
    "model_21": "Time- and region-varying\nintercepts and slopes TL",
    "model_22": "Random walk priors TL",
}


def main() -> None:
    data = cb.read_data(ROOT / "data" / "log_df.csv")
    rows = []
    for model_key, label in MODEL_NAMES.items():
        _, trace = cb.urban_scaling(
            df=data,
            dependent_var="Murder",
            model_key=model_key,
            save_path=str(RESULTS / model_key / "Murder"),
            loadtrace=True,
        )
        waic, standard_error, effective_parameters = trace["waic"]
        rows.append((label.replace(" TL", ""), waic, standard_error, effective_parameters))

    comparison = pd.DataFrame(rows, columns=["model", "waic", "se", "p_waic"])
    # Supplementary Figure S2 compares the Taylor-law model family only.
    comparison = comparison.iloc[11:].sort_values("waic", ascending=True)

    figure, axis = plt.subplots(figsize=(7, 10))
    y = np.arange(len(comparison))
    axis.errorbar(
        comparison["waic"], y, xerr=comparison["se"], fmt="none",
        ecolor="#e31a1c", elinewidth=2.5, capsize=5, capthick=2,
    )
    axis.plot(comparison["waic"], y, "o", color="#e31a1c", markersize=7)
    axis.axvline(comparison["waic"].iloc[-1], color="gray", linewidth=1.5, zorder=-1)
    axis.set_yticks(y, comparison["model"])
    axis.set_title("Model selection")
    axis.set_xlabel("WAIC")
    figure.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
